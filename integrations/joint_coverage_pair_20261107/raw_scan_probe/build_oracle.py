#!/usr/bin/env python3
"""Compile exact vendor decoder/validity/rebin bodies with non-ROS fixture types.

This builds an offline library only. It never loads or starts the live driver.
"""
import hashlib,json,pathlib,subprocess
P=pathlib.Path(__file__).resolve().parent
V=pathlib.Path('/home/ouc/scout_nav/src/lslidar_n301/lslidar_driver')

def body(text,name):
    pos=text.index('    '+name);start=text.index('{',pos);depth=1;end=start+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[pos:end]

def build():
    source=(V/'src/lslidar_x10_driver.cpp').read_text()
    expected='6a60a2aef316b285aacf1908d8f58004d11157cdde9c1a9056e890641938ae94'
    assert hashlib.sha256(source.encode()).hexdigest()==expected,'Vendor source changed; review before building'
    headers={'include/lslidar_driver/lslidar_x10_driver.hpp':'76cdbdc4971d7c02d831ffd64d6d669c73d687f891819b273f848bd8f946c879','include/lslidar_driver/lslidar_driver.hpp':'6ab686ae9b4ef91725699f8dea3abc6cae3d61bae0d3126a365fe54122985764'}
    for name,digest in headers.items():assert hashlib.sha256((V/name).read_bytes()).hexdigest()==digest,'Vendor header changed: '+name
    funcs=[body(source,n) for n in ['void LslidarX10Driver::decodePacketN301_1_7(', 'bool LslidarX10Driver::isPointValid(', 'void LslidarX10Driver::pointcloudToLaserscan(']]
    pre=r'''
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
struct PointCloud2 { int header=0; std::vector<float> xyzi; };
struct LaserScan {int header=0; std::string unused; float angle_min,angle_max,time_increment,range_min,range_max,angle_increment; std::vector<float> ranges,intensities;};
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
'''
    # The real PointCloud2 header has frame_id; provide that field, without changing bodies.
    pre=pre.replace('struct PointCloud2 { int header=0;', 'struct Header { std::string frame_id; }; struct PointCloud2 { Header header;').replace('struct LaserScan {int header=0;', 'struct LaserScan {Header header;')
    post=r'''
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
'''
    cpp=P/'vendor_oracle.cpp';cpp.write_text(source[:source.index('#include')]+pre+'\n'.join(funcs)+post)
    command=['/usr/bin/c++','-std=c++11','-O2','-shared','-fPIC',str(cpp),'-o',str(P/'vendor_oracle.so')]
    subprocess.run(command,check=True)
    receipt={'scope':'offline fixture library; original vendor decoder/validity/rebin bodies compiled with minimal non-ROS types, no driver executed','vendor_source_sha256':expected,'vendor_header_sha256':headers,'function_body_sha256':[hashlib.sha256(f.encode()).hexdigest() for f in funcs],'command':command,'library_sha256':hashlib.sha256((P/'vendor_oracle.so').read_bytes()).hexdigest()}
    (P/'oracle_build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
if __name__=='__main__':print(json.dumps(build(),indent=2))
