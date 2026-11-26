#include <costmap_2d/cost_values.h>
#include <costmap_2d/costmap_2d.h>
#include <explore/frontier_search.h>
#include <geometry_msgs/Point.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
  unsigned int width, height, robot_col, robot_row;
  double resolution;
  if (!(std::cin >> width >> height >> resolution >> robot_col >> robot_row) ||
      width == 0 || height == 0 || resolution <= 0.0 ||
      robot_col >= width || robot_row >= height) {
    std::cerr << "Invalid grid or robot position\n";
    return 2;
  }
  costmap_2d::Costmap2D map(width, height, resolution, 0.0, 0.0);
  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      int cell;
      if (!(std::cin >> cell) || (cell != -1 && cell != 0 && cell != 1)) {
        std::cerr << "Unknown grid encoding\n";
        return 2;
      }
      map.setCost(col, row, cell == -1 ? costmap_2d::NO_INFORMATION :
                             cell == 0 ? costmap_2d::FREE_SPACE : costmap_2d::LETHAL_OBSTACLE);
    }
  }
  geometry_msgs::Point robot;
  map.mapToWorld(robot_col, robot_row, robot.x, robot.y);
  // The pinned upstream explore_costmap.launch settings.
  frontier_exploration::FrontierSearch search(&map, 3.0, 1.0, 0.5);
  std::cout << std::setprecision(17);
  for (const auto& frontier : search.searchFrom(robot)) {
    std::cout << frontier.cost << " " << frontier.centroid.y << " "
              << frontier.centroid.x;
    std::vector<geometry_msgs::Point> cells{frontier.initial};
    cells.insert(cells.end(), frontier.points.begin(), frontier.points.end());
    for (const auto& point : cells) {
      unsigned int col, row;
      if (!map.worldToMap(point.x, point.y, col, row)) {
        std::cerr << "Native frontier outside grid\n";
        return 3;
      }
      std::cout << " " << row << " " << col;
    }
    std::cout << "\n";
  }
}
