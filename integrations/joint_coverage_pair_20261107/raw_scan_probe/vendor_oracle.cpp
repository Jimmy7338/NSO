/******************************************************************************
 * This file is part of lslidar_driver.
 *
 * Copyright 2022 LeiShen Intelligent Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/


// Original function bodies: LeiShen Intelligent, Apache-2.0; vendor source SHA in oracle_build.json.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#define ROS_DEBUG(...) ((void)0)
namespace lslidar_msgs { struct LslidarPacket { uint8_t data[1206]; }; using LslidarPacketPtr=std::shared_ptr<LslidarPacket>; }
namespace sensor_msgs {
struct Header { std::string frame_id; }; struct PointCloud2 { Header header; std::vector<float> xyzi; };
struct LaserScan {Header header; std::string unused; float angle_min,angle_max,time_increment,range_min,range_max,angle_increment; std::vector<float> ranges,intensities;};
template<class T> struct PointCloud2ConstIterator {
 const PointCloud2 *c; size_t i; int field;
 PointCloud2ConstIterator(const PointCloud2& v,const char *name):c(&v),i(0),field(std::string(name)=="x"?0:std::string(name)=="y"?1:std::string(name)=="z"?2:3){}
 PointCloud2ConstIterator end() const {auto v=*this;v.i=c->xyzi.size()/4;return v;}
 bool operator!=(const PointCloud2ConstIterator& v)const{return i!=v.i;}
 void operator++(){++i;} T operator*()const{return c->xyzi[4*i+field];}
}; }
namespace lslidar_driver {
static const int BLOCKS_PER_PACKET_N301=12,SIZE_BLOCK_N301=100,SCANS_PER_BLOCK_N301_1_7=31,RAW_SCAN_SIZE_N301=3,FIRING_TOFFSET_N301_1_7=30;
static const float DISTANCE_RESOLUTION_N301_1_7=.004f,INITIAL_RANGE=std::numeric_limits<float>::infinity(),INITIAL_INTENSITIE=std::numeric_limits<float>::quiet_NaN();
struct FiringX10 {int azimuth;float distance,intensity;};
class LslidarX10Driver {public:
 int actual_points=0,packet_points_max=360;uint16_t points_size=2000;double min_range=.15,max_range=25;
 int angle_able_min=0,angle_able_max=36000;bool use_high_precision=false;std::vector<FiringX10> points;
 void decodePacketN301_1_7(lslidar_msgs::LslidarPacketPtr&);
 bool isPointValid(int)const;
 void pointcloudToLaserscan(const sensor_msgs::PointCloud2&,sensor_msgs::LaserScan&);
};
    void LslidarX10Driver::decodePacketN301_1_7(lslidar_msgs::LslidarPacketPtr &packet) {
        uint16_t firing_azimuth[BLOCKS_PER_PACKET_N301];

        for (size_t b_idx = 0; b_idx < BLOCKS_PER_PACKET_N301; ++b_idx) {
            firing_azimuth[b_idx] = (packet->data[b_idx * SIZE_BLOCK_N301 + 2] + (packet->data[b_idx * SIZE_BLOCK_N301 + 3] << 8)) % 36000;
        }

        actual_points = packet_points_max;

        points.clear();
        points.reserve(actual_points);

        for (size_t block_idx = 0; block_idx < BLOCKS_PER_PACKET_N301; ++block_idx) {
            int32_t azimuth_diff_b = 0;
            if (block_idx < BLOCKS_PER_PACKET_N301 - 1) {
                azimuth_diff_b = firing_azimuth[block_idx + 1] - firing_azimuth[block_idx];
            } else {
                azimuth_diff_b = firing_azimuth[block_idx] - firing_azimuth[block_idx - 1];
            }
            azimuth_diff_b = azimuth_diff_b < 0 ? azimuth_diff_b + 36000 : azimuth_diff_b;

            // 每一块30个点
            for (size_t scan_fir_idx = 0; scan_fir_idx < SCANS_PER_BLOCK_N301_1_7; ++scan_fir_idx) {
                if (scan_fir_idx == 15) continue;   // 跳过非点信息 每块第 49 50 51 byte为 年 月 日
                size_t byte_idx = RAW_SCAN_SIZE_N301 * scan_fir_idx;
                size_t idx = scan_fir_idx;
                if (scan_fir_idx > 15) idx -= 1;

                FiringX10 point;
                point.azimuth = firing_azimuth[block_idx] + idx * azimuth_diff_b / FIRING_TOFFSET_N301_1_7;
                point.azimuth = point.azimuth % 36000;
                point.distance = static_cast<float>((packet->data[block_idx * SIZE_BLOCK_N301 + byte_idx + 4] + 
                                                    (packet->data[block_idx * SIZE_BLOCK_N301 + byte_idx + 5] << 8)) * DISTANCE_RESOLUTION_N301_1_7);
                point.intensity = static_cast<float>(packet->data[block_idx * SIZE_BLOCK_N301 + byte_idx + 6]);

                points.push_back(point);
            }
        }
    }
    bool LslidarX10Driver::isPointValid(const int fir_idx) const {
        if (!(points[fir_idx].distance >= min_range && points[fir_idx].distance <= max_range)) return false;

        if (angle_able_max > 36000) {
            if ((points[fir_idx].azimuth > (angle_able_max - 36000)) && points[fir_idx].azimuth < angle_able_min)  return false;
        } else {
            if ((points[fir_idx].azimuth > angle_able_max) || points[fir_idx].azimuth < angle_able_min) return false;
        }

        return true;
    }
    void LslidarX10Driver::pointcloudToLaserscan(const sensor_msgs::PointCloud2 &cloud_msg,
                                             sensor_msgs::LaserScan &output_scan) {
        output_scan.header = cloud_msg.header;
        output_scan.header.frame_id = cloud_msg.header.frame_id;
        output_scan.angle_min = -M_PI;
        output_scan.angle_max = M_PI;
        output_scan.time_increment = 0.0; 
        output_scan.range_min = min_range; 
        output_scan.range_max = max_range; 
        output_scan.angle_increment = (output_scan.angle_max - output_scan.angle_min) / points_size;

        // 使用高精度模式 角度分辨率减小10倍 数据量增加10倍 保证laserscan数据与点云一致
        // N10等点数较少雷达使用实际角度分辨率时，部分laserscan数据会与点云有所偏差
        if (use_high_precision) output_scan.angle_increment *= 0.1;

        uint16_t ranges_size = std::ceil((output_scan.angle_max - output_scan.angle_min) / output_scan.angle_increment);
        output_scan.ranges.assign(ranges_size, INITIAL_RANGE); 
        output_scan.intensities.assign(ranges_size, INITIAL_INTENSITIE);

        for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_msg, "x"), iter_y(cloud_msg, "y"),
                iter_z(cloud_msg, "z"), iter_intensity(cloud_msg, "intensity");
            iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {

            if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)) {
                ROS_DEBUG("rejected for nan in point(%f, %f, %f)\n", *iter_x, *iter_y, *iter_z);
                continue;
            }

            double range = hypot(*iter_x, *iter_y);
            if (range < min_range || range > max_range) {
                ROS_DEBUG("rejected for range %f not in range [%f, %f]. Point: (%f, %f, %f)",
                        range, min_range, max_range, *iter_x, *iter_y, *iter_z);
                continue;
            }

            double angle = atan2(*iter_y, *iter_x);
            if (angle < output_scan.angle_min || angle > output_scan.angle_max) {
                ROS_DEBUG("rejected for angle %f not in range [%f, %f]\n", angle,
                        output_scan.angle_min, output_scan.angle_max);
                continue;
            }

            int index = static_cast<int>((angle - output_scan.angle_min) / output_scan.angle_increment);
            if (index >= 0 && index < ranges_size && range < output_scan.ranges[index]) {
                output_scan.ranges[index] = range;
                output_scan.intensities[index] = *iter_intensity;
            }
        }
    }
}
extern "C" int decode(const unsigned char *data,int *az,float *range,float *intensity){
 auto p=std::make_shared<lslidar_msgs::LslidarPacket>();memcpy(p->data,data,1206);
 for(int b=0;b<12;++b)if(data[b*100]!=255||data[b*100+1]!=238)return -1;
 lslidar_driver::LslidarX10Driver d;d.decodePacketN301_1_7(p);
 for(int i=0;i<360;++i){az[i]=d.points[i].azimuth;range[i]=d.points[i].distance;intensity[i]=d.points[i].intensity;}return 360;
}
extern "C" int rebin(const int *az,const float *range,const float *intensity,int n,float *ranges,float *ints,float *xyzi){
 lslidar_driver::LslidarX10Driver d;sensor_msgs::PointCloud2 c;
 for(int i=0;i<n;++i){d.points.push_back({az[i],range[i],intensity[i]});if(!d.isPointValid(i))continue;
 // Exact vendor initAngleConfig and poll float geometry expressions, DEG_TO_RAD=0.017453f.
 float angle=static_cast<float>(az[i])*.01f*.017453f;
 float x_coord=range[i]*cosf(angle),y_coord=-range[i]*sinf(angle);
 c.xyzi.insert(c.xyzi.end(),{x_coord,y_coord,0.f,intensity[i]});}
 sensor_msgs::LaserScan s;d.pointcloudToLaserscan(c,s);
 if(s.ranges.size()!=2000)return -1;
 memcpy(ranges,s.ranges.data(),8000);memcpy(ints,s.intensities.data(),8000);
 memcpy(xyzi,c.xyzi.data(),c.xyzi.size()*4);return c.xyzi.size()/4;
}
