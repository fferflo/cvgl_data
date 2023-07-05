#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <xtensor-python/pytensor.hpp>
#include <pybind11/pybind11.h>
#include <cosy/python.h>
#include <pybind11/stl.h>
#include <sstream>
#include <pybind11/stl_bind.h>

#include "frame.h"

namespace py = pybind11;

PYBIND11_MODULE(backend, m)
{
  py::class_<cvgl_data::ToStringHelper, std::shared_ptr<cvgl_data::ToStringHelper>>(m, "ToStringHelper", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::ToStringHelper& self){
      return self.to_string();
    })
  ;

  py::class_<cvgl_data::Data, std::shared_ptr<cvgl_data::Data>>(m, "Data", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Data& self){
      return self.to_string();
    })
    .def_property_readonly("timestamp", &cvgl_data::Data::get_timestamp)
  ;
  py::class_<cvgl_data::Loader, std::shared_ptr<cvgl_data::Loader>>(m, "Loader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Loader& self){
      return self.to_string();
    })
    .def_property_readonly("timestamps", [](cvgl_data::Loader& self){
      const auto& timestamps = self.get_timesequence().get_timestamps();
      if (timestamps.shape()[0] == 0)
      {
        throw std::invalid_argument("No timestamps available");
      }
      return timestamps;
    })
  ;

  py::class_<cvgl_data::NamedData, std::shared_ptr<cvgl_data::NamedData>, cvgl_data::Data>(m, "NamedData", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::NamedData& self){
      return self.to_string();
    })
    .def("__len__", [](cvgl_data::NamedData& self){
      return self.get_all().size();
    })
    .def("__getitem__", &cvgl_data::NamedData::get)
    .def("__contains__", [](cvgl_data::NamedData& self, std::string name){
      return self.get_all().find(name) != self.get_all().end();
    })
    .def("keys", [](cvgl_data::NamedData& self){
      std::vector<std::string> result;
      for (auto pair : self.get_all())
      {
        result.push_back(pair.first);
      }
      return result;
    })
    .def("values", [](cvgl_data::NamedData& self){
      std::vector<std::shared_ptr<cvgl_data::Data>> result;
      for (auto pair : self.get_all())
      {
        result.push_back(pair.second);
      }
      return result;
    })
    .def(py::pickle(
        [](const cvgl_data::NamedData& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::NamedData::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::NamedDataLoader, std::shared_ptr<cvgl_data::NamedDataLoader>, cvgl_data::Loader>(m, "NamedDataLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::NamedDataLoader& self){
      return self.to_string();
    })
    .def("__len__", [](cvgl_data::NamedDataLoader& self){
      return self.get_loaders().size();
    })
    .def("__getitem__", &cvgl_data::NamedDataLoader::get)
    .def("load", [](cvgl_data::NamedDataLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
    .def("keys", [](cvgl_data::NamedDataLoader& self){
      std::vector<std::string> result;
      for (auto pair : self.get_loaders())
      {
        result.push_back(pair.first);
      }
      return result;
    })
    .def("values", [](cvgl_data::NamedDataLoader& self){
      std::vector<std::shared_ptr<cvgl_data::Loader>> result;
      for (auto pair : self.get_loaders())
      {
        result.push_back(pair.second);
      }
      return result;
    })
  ;

  py::class_<cvgl_data::EgoToWorld, std::shared_ptr<cvgl_data::EgoToWorld>, cvgl_data::Data>(m, "EgoToWorld", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::EgoToWorld& self){
      return self.to_string();
    })
    .def_property_readonly("transform", &cvgl_data::EgoToWorld::get_transform)
    .def(py::pickle(
        [](const cvgl_data::EgoToWorld& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::EgoToWorld::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::EgoToWorldLoader, std::shared_ptr<cvgl_data::EgoToWorldLoader>, cvgl_data::Loader>(m, "EgoToWorldLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::EgoToWorldLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::EgoToWorldLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
  ;

  py::class_<cvgl_data::GeoPose, std::shared_ptr<cvgl_data::GeoPose>, cvgl_data::Data>(m, "GeoPose", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::GeoPose& self){
      return self.to_string();
    })
    .def_property_readonly("latlon", &cvgl_data::GeoPose::get_latlon)
    .def_property_readonly("bearing", &cvgl_data::GeoPose::get_bearing)
    .def(py::pickle(
        [](const cvgl_data::GeoPose& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::GeoPose::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::GeoPoseLoader, std::shared_ptr<cvgl_data::GeoPoseLoader>, cvgl_data::Loader>(m, "GeoPoseLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::GeoPoseLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::GeoPoseLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
  ;

  py::class_<cvgl_data::OutlierScore, std::shared_ptr<cvgl_data::OutlierScore>, cvgl_data::Data>(m, "OutlierScore", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::OutlierScore& self){
      return self.to_string();
    })
    .def_property_readonly("score", &cvgl_data::OutlierScore::get_score)
    .def(py::pickle(
        [](const cvgl_data::OutlierScore& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::OutlierScore::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::OutlierScoreLoader, std::shared_ptr<cvgl_data::OutlierScoreLoader>, cvgl_data::Loader>(m, "OutlierScoreLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::OutlierScoreLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::OutlierScoreLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
  ;

  auto cam_ops = m.def_submodule("cam_ops");

  py::class_<cvgl_data::cam_ops::Op, std::shared_ptr<cvgl_data::cam_ops::Op>>(cam_ops, "Op", py::dynamic_attr());
  py::class_<cvgl_data::cam_ops::Tile, std::shared_ptr<cvgl_data::cam_ops::Tile>, cvgl_data::cam_ops::Op>(cam_ops, "Tile", py::dynamic_attr())
    .def(py::init([](xti::vec2u tile_shape, std::optional<xti::vec2u> tile_crop_margin){
        return cvgl_data::cam_ops::Tile(tile_shape, tile_crop_margin);
      }),
      py::arg("tile_shape"),
      py::arg("tile_crop_margin") = std::optional<xti::vec2u>()
    )
  ;
  py::class_<cvgl_data::cam_ops::Resize, std::shared_ptr<cvgl_data::cam_ops::Resize>, cvgl_data::cam_ops::Op>(cam_ops, "Resize", py::dynamic_attr());
  py::class_<cvgl_data::cam_ops::Filter, std::shared_ptr<cvgl_data::cam_ops::Filter>, cvgl_data::cam_ops::Op>(cam_ops, "Filter", py::dynamic_attr());
  py::class_<cvgl_data::cam_ops::Homography, std::shared_ptr<cvgl_data::cam_ops::Homography>, cvgl_data::cam_ops::Op>(cam_ops, "Homography", py::dynamic_attr());

  cam_ops
    .def("ResizeBy", [](float scale){
        return cvgl_data::cam_ops::Resize([=](const cvgl_data::CameraLoader& camera){return scale;});
      },
      py::arg("scale")
    )
    .def("ResizeToFocalLength", [](float focal_length){
        return cvgl_data::cam_ops::Resize([=](const cvgl_data::CameraLoader& camera){
          auto intr = camera.get_intr().get_matrix();
          float src_focal_length = 0.5 * (intr(0, 0) + intr(1, 1));
          return focal_length / src_focal_length;
        });
      },
      py::arg("scale")
    )
    .def("KeepNames", [](std::vector<std::string> names){
        return cvgl_data::cam_ops::Filter([=](const cvgl_data::CameraLoader& camera){
          return std::find(names.begin(), names.end(), camera.get_name()) != names.end();
        });
      },
      py::arg("names")
    )
    .def("None", [](){
        return cvgl_data::cam_ops::Filter([=](const cvgl_data::CameraLoader& camera){
          return false;
        });
      }
    )
    .def("ConstantHomography", [](cosy::Rigid<float, 3> newcam_to_oldcam){
        return cvgl_data::cam_ops::Homography([=](cosy::Rigid<float, 3> oldcam_to_ego, std::optional<cosy::Rigid<float, 3>> ego_to_world){
          return oldcam_to_ego * newcam_to_oldcam;
        });
      },
      py::arg("newcam_to_oldcam")
    )
    .def("AlignWithUpVectorHomography", [](std::string reference_frame){
        bool use_world_ref;
        if (reference_frame == "world")
        {
          use_world_ref = true;
        }
        else if (reference_frame == "ego")
        {
          use_world_ref = false;
        }
        else
        {
          throw std::invalid_argument("reference_frame must be either 'world' or 'ego'");
        }
        return cvgl_data::cam_ops::Homography([=](cosy::Rigid<float, 3> cam_to_ego, std::optional<cosy::Rigid<float, 3>> ego_to_world){
          cosy::Rigid<float, 3> cam_to_ref;
          if (use_world_ref)
          {
            if (!ego_to_world)
            {
              throw std::invalid_argument("This homography requires ego_to_world to be passed");
            }
            cam_to_ref = ego_to_world.value() * cam_to_ego;
          }
          else
          {
            cam_to_ref = cam_to_ego;
          }

          xti::vec3f cam_up = xti::vec3f({0, -1, 0});
          cam_up = cam_up / xt::linalg::norm(cam_up);
          xti::vec3f ref_up = xt::linalg::dot(cam_to_ref.inverse().get_rotation(), xti::vec3f({0, 0, 1}));
          ref_up = ref_up / xt::linalg::norm(ref_up);
          float angle = std::acos(xt::sum(cam_up * ref_up)());
          xti::vec3f normal = xt::linalg::cross(cam_up, ref_up);
          if (std::abs(angle) > 1e-6 && xt::linalg::norm(normal) > 1e-6)
          {
            cosy::Rigid<float, 3> newcam_to_oldcam(cosy::axisangle_to_rotation_matrix(normal, angle), xti::vec3f({0, 0, 0}));
            return cam_to_ego * newcam_to_oldcam;
          }
          else
          {
            return cam_to_ego;
          }
        });
      },
      py::arg("reference_frame")
    )
  ;

  py::class_<cvgl_data::Camera, std::shared_ptr<cvgl_data::Camera>, cvgl_data::Data>(m, "Camera", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Camera& self){
      return self.to_string();
    })
    .def_property("image", &cvgl_data::Camera::get_image, &cvgl_data::Camera::set_image)
    .def_property_readonly("cam_to_ego", &cvgl_data::Camera::get_cam_to_ego)
    .def_property_readonly("intr", &cvgl_data::Camera::get_projection)
    .def_property_readonly("name", &cvgl_data::Camera::get_name)
    .def(py::pickle(
        [](const cvgl_data::Camera& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::Camera::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::CameraLoader, std::shared_ptr<cvgl_data::CameraLoader>, cvgl_data::Loader>(m, "CameraLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::CameraLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::CameraLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
    .def_property_readonly("resolution", &cvgl_data::CameraLoader::get_resolution)
    .def_property_readonly("name", &cvgl_data::CameraLoader::get_name)
    .def_property_readonly("intr", &cvgl_data::CameraLoader::get_intr)
  ;

  auto lidar_ops = m.def_submodule("lidar_ops");

  py::class_<cvgl_data::lidar_ops::Op, std::shared_ptr<cvgl_data::lidar_ops::Op>>(lidar_ops, "Op", py::dynamic_attr());
  py::class_<cvgl_data::lidar_ops::Filter, std::shared_ptr<cvgl_data::lidar_ops::Filter>, cvgl_data::lidar_ops::Op>(lidar_ops, "Filter", py::dynamic_attr());

  lidar_ops
    .def("KeepNames", [](std::vector<std::string> names){
        return cvgl_data::lidar_ops::Filter([=](const cvgl_data::LidarLoader& lidar){
          return std::find(names.begin(), names.end(), lidar.get_name()) != names.end();
        });
      },
      py::arg("names")
    )
    .def("None", [](){
        return cvgl_data::lidar_ops::Filter([=](const cvgl_data::LidarLoader& lidar){
          return false;
        });
      }
    )
  ;

  py::class_<cvgl_data::Lidar, std::shared_ptr<cvgl_data::Lidar>, cvgl_data::Data>(m, "Lidar", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Lidar& self){
      return self.to_string();
    })
    .def_property_readonly("points", &cvgl_data::Lidar::get_points)
    .def_property_readonly("name", &cvgl_data::Lidar::get_name)
    .def(py::pickle(
        [](const cvgl_data::Lidar& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::Lidar::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::LidarLoader, std::shared_ptr<cvgl_data::LidarLoader>, cvgl_data::Loader>(m, "LidarLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::LidarLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::LidarLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
    .def_property_readonly("name", &cvgl_data::LidarLoader::get_name)
  ;

  py::class_<cvgl_data::Lidars, std::shared_ptr<cvgl_data::Lidars>, cvgl_data::NamedData>(m, "Lidars", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Lidars& self){
      return self.to_string();
    })
    .def_property_readonly("points", [](cvgl_data::Lidars& self){
        py::gil_scoped_release gil;
        return self.get_points();
      }
    )
    .def(py::pickle(
        [](const cvgl_data::Lidars& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::Lidars::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::LidarsLoader, std::shared_ptr<cvgl_data::LidarsLoader>, cvgl_data::NamedDataLoader>(m, "LidarsLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::LidarsLoader& self){
      return self.to_string();
    })
  ;

  py::class_<cvgl_data::Map, std::shared_ptr<cvgl_data::Map>, cvgl_data::Data>(m, "Map", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Map& self){
      return self.to_string();
    })
    .def_property("image", &cvgl_data::Map::get_image, &cvgl_data::Map::set_image)
    .def_property_readonly("name", &cvgl_data::Map::get_name)
    .def_property_readonly("meters_per_pixel", &cvgl_data::Map::get_meters_per_pixel)
    .def(py::pickle(
        [](const cvgl_data::Map& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::Map::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::MapLoader, std::shared_ptr<cvgl_data::MapLoader>, cvgl_data::Loader>(m, "MapLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::MapLoader& self){
      return self.to_string();
    })
    .def("load", [](cvgl_data::MapLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
    .def_property_readonly("name", &cvgl_data::MapLoader::get_name)
    .def_property_readonly("resolution", &cvgl_data::MapLoader::get_resolution)
  ;

  py::class_<cvgl_data::Frame, std::shared_ptr<cvgl_data::Frame>, cvgl_data::NamedData>(m, "Frame", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::Frame& self){
      return self.to_string();
    })
    // .def("move_ego", [](cvgl_data::Frame& self, cosy::Rigid<float, 3> oldego_to_newego){
    //     py::gil_scoped_release gil;
    //     return self.move_ego(oldego_to_newego);
    //   }
    // )
    .def_property_readonly("scene_name", &cvgl_data::Frame::get_scene_name)
    .def_property_readonly("location", &cvgl_data::Frame::get_location)
    .def_property_readonly("dataset", &cvgl_data::Frame::get_dataset)
    .def_property_readonly("name", &cvgl_data::Frame::get_name)
    .def(py::pickle(
        [](const cvgl_data::Frame& obj) {
          return obj.pickle();
        },
        [](py::tuple t) {
          return cvgl_data::Frame::unpickle(t);
        }
    ))
  ;
  py::class_<cvgl_data::FrameLoader, std::shared_ptr<cvgl_data::FrameLoader>, cvgl_data::NamedDataLoader>(m, "FrameLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::FrameLoader& self){
      return self.to_string();
    })
    .def(py::init([](std::string path, std::vector<std::shared_ptr<cvgl_data::cam_ops::Op>> cam_ops, std::vector<std::shared_ptr<cvgl_data::lidar_ops::Op>> lidar_ops, std::vector<std::string> updates){
        py::gil_scoped_release gil;
        std::vector<std::filesystem::path> std_updates;
        for (std::string update : updates)
        {
          std_updates.push_back(std::filesystem::path(update));
        }

        return cvgl_data::FrameLoader::construct(path, cam_ops, lidar_ops, std_updates);
      }),
      py::arg("path"),
      py::arg("cam_ops") = std::vector<std::shared_ptr<cvgl_data::cam_ops::Op>>(),
      py::arg("lidar_ops") = std::vector<std::shared_ptr<cvgl_data::lidar_ops::Op>>(),
      py::arg("updates") = std::vector<std::string>()
    )
    .def("load", [](cvgl_data::FrameLoader& self, uint64_t timestamp){
        py::gil_scoped_release gil;
        return self.load(timestamp);
      }
    )
    .def_property_readonly("scene_name", &cvgl_data::FrameLoader::get_scene_name)
    .def_property_readonly("location", &cvgl_data::FrameLoader::get_location)
    .def_property_readonly("dataset", &cvgl_data::FrameLoader::get_dataset)
  ;

  py::class_<cvgl_data::TiledWebMapsLoader, std::shared_ptr<cvgl_data::TiledWebMapsLoader>>(m, "TiledWebMapsLoader", py::dynamic_attr())
    .def("__repr__", [](cvgl_data::TiledWebMapsLoader& self){
      return self.to_string();
    })
    .def(py::init([](std::shared_ptr<tiledwebmaps::TileLoader> tileloader, std::string name, size_t zoom){
        py::gil_scoped_release gil;
        return std::make_shared<cvgl_data::TiledWebMapsLoader>(tileloader, name, zoom);
      }),
      py::arg("tileloader"),
      py::arg("name"),
      py::arg("zoom")
    )
    .def("load", [](cvgl_data::TiledWebMapsLoader& self, xti::vec2f latlon, float bearing, float meters_per_pixel, xti::vec2s shape, std::string location){
        py::gil_scoped_release gil;
        return self.load(latlon, bearing, meters_per_pixel, shape, location);
      },
      py::arg("latlon"),
      py::arg("bearing"),
      py::arg("meters_per_pixel"),
      py::arg("shape"),
      py::arg("location") = "unknown-location"
    )
    .def_property_readonly("name", &cvgl_data::TiledWebMapsLoader::get_name)
    .def_property_readonly("zoom", &cvgl_data::TiledWebMapsLoader::get_zoom)
    .def_property_readonly("tileloader", &cvgl_data::TiledWebMapsLoader::get_tileloader)
  ;
}
