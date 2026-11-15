// Exact counts for Python-supplied original ray templates. No ROS/SDK/I/O.
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <new>
extern "C" int nso_batch_visibility_counts(
 const std::uint8_t* occupied,const std::uint8_t* query,
 std::int64_t height,std::int64_t width,const std::int64_t* positions,
 std::int64_t count,const std::int64_t* offsets,const std::uint8_t* valid,
 std::int64_t rays,std::int64_t steps,std::int64_t* output) noexcept {
 if(!occupied||!query||!positions||!offsets||!valid||!output||height<=0||width<=0||count<0||rays<=0||steps<=0) return -1;
 if(height>std::numeric_limits<std::int64_t>::max()/width) return -1;
 try {
  std::vector<std::int32_t> stamps(static_cast<std::size_t>(height*width),0);
  std::int32_t epoch=0;
  for(std::int64_t k=0;k<count;++k) {
   if(epoch==std::numeric_limits<std::int32_t>::max()) {std::fill(stamps.begin(),stamps.end(),0);epoch=0;}
   ++epoch;std::int64_t total=0;
   const auto pr=positions[k*2],pc=positions[k*2+1];
   for(std::int64_t ray=0;ray<rays;++ray) {
    const auto begin=ray*steps;
    for(std::int64_t step=0;step<steps;++step) {
     const auto index=begin+step;
     if(!valid[index]) break;
     const auto row=pr+offsets[index*2],col=pc+offsets[index*2+1];
     if(row<0||row>=height||col<0||col>=width) break;
     const auto cell=static_cast<std::size_t>(row*width+col);
     if(stamps[cell]!=epoch) {stamps[cell]=epoch;if(query[cell]) ++total;}
     if(occupied[cell]) break;
    }
   }
   output[k]=total;
  }
  return 0;
 } catch(const std::bad_alloc&) {return -2;} catch(...) {return -3;}
}
