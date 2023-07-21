#pragma once

#include <yaml-cpp/yaml.h>
#include <cosy/affine.h>
#include <cosy/proj.h>
#include <xtensor/xtensor.hpp>
#include <xti/opencv.h>
#include <xti/util.h>
#include <xtensor-blas/xlinalg.hpp>
#include <memory>
#include <filesystem>
#include <xtensor-io/xnpz.hpp>
#include <tiledwebmaps/tiledwebmaps.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <functional>
#include <sstream>

namespace cvgl_data {

class Yaml
{
public:
  Yaml(std::filesystem::path file, bool verbose)
    : m_file(file)
    , m_parent_keys("")
  {
    if (verbose)
    {
      std::cout << "cvgl_data: Loading yaml file " << file.string() << std::flush;
    }
    m_node = YAML::LoadFile(file.string());
    if (verbose)
    {
      std::cout << " done" << std::endl;
    }
  }

  template <typename T>
  Yaml operator[](T key) const
  {
    if (!m_node[key])
    {
      throw std::runtime_error(XTI_TO_STRING("Yaml file " << m_file.string() << " does not contain key \"" << key << "\" for parents \"" << m_parent_keys << "\""));
    }
    return Yaml(m_node[key], XTI_TO_STRING(m_parent_keys << "." << key));
  }

  template <typename T>
  T as() const
  {
    try
    {
      return m_node.as<T>();
    }
    catch (const YAML::Exception& e)
    {
      throw std::runtime_error(XTI_TO_STRING("Failed to convert object at \"" << m_parent_keys << "\" to type \"" << typeid(T).name() << "\": " << e.what()));
    }
  }

private:
  Yaml(YAML::Node node, std::string parent_keys)
    : m_node(node)
  {
  }

  std::filesystem::path m_file;
  YAML::Node m_node;
  std::string m_parent_keys;
};

auto load_npz(std::filesystem::path path, bool verbose)
{
  if (verbose)
  {
    std::cout << "cvgl_data: Loading npz file " << path.string() << std::flush;
  }
  auto npz = xt::load_npz(path);
  if (verbose)
  {
    std::cout << " done" << std::endl;
  }
  return npz;
}

cv::Mat imread(std::filesystem::path path, bool verbose)
{
  if (verbose)
  {
    std::cout << "cvgl_data: Loading image " << path.string() << std::flush;
  }
  cv::Mat image = tiledwebmaps::safe_imread(path.string());
  if (verbose)
  {
    std::cout << " done" << std::endl;
  }
  return image;
}

class FrameLoader;

cosy::Rigid<float, 3> yaml_to_transform(const Yaml& node)
{
  auto translation_node = node["translation"];
  auto rotation_node = node["rotation"];

  cosy::Rigid<float, 3> result;
  for (size_t r = 0; r < 3; r++)
  {
    result.get_translation()(r) = translation_node[r].as<float>();
    for (size_t c = 0; c < 3; c++)
    {
      result.get_rotation()(r, c) = rotation_node[r][c].as<float>();
    }
  }
  return result;
}

xti::mat3f yaml_to_projection(const Yaml& node)
{
  xti::mat3f result;
  for (size_t r = 0; r < 3; r++)
  {
    for (size_t c = 0; c < 3; c++)
    {
      result(r, c) = node[r][c].as<float>();
    }
  }
  return result;
}

template <typename TElementType, size_t TRank, typename TInput>
xt::xtensor<TElementType, TRank> load_from_npz_int(TInput&& input)
{
  xt::xtensor<TElementType, TRank> output;
  if (input.m_typestring == "<u4")
  {
    output = input.template cast<uint32_t>();
  }
  else if (input.m_typestring == "<i4")
  {
    output = input.template cast<int32_t>();
  }
  else if (input.m_typestring == "<u8")
  {
    output = input.template cast<uint64_t>();
  }
  else if (input.m_typestring == "<i8")
  {
    output = input.template cast<int64_t>();
  }
  else
  {
    throw std::runtime_error(XTI_TO_STRING("Npz file has invalid datatype " << input.m_typestring));
  }
  return output;
}

template <typename TElementType, size_t TRank, typename TInput>
xt::xtensor<TElementType, TRank> load_from_npz_float(TInput&& input)
{
  xt::xtensor<TElementType, TRank> output;
  if (input.m_typestring == "<f4")
  {
    output = input.template cast<float>();
  }
  else if (input.m_typestring == "<f8")
  {
    output = input.template cast<double>();
  }
  else
  {
    throw std::runtime_error(XTI_TO_STRING("Npz file has invalid datatype " << input.m_typestring));
  }
  return output;
}

static const std::string INDENT = "  ";

class ToStringHelper
{
public:
  virtual ~ToStringHelper() = default;

  std::string to_string(std::string indent = "")
  {
    std::stringstream str;
    str << "{\n";

    std::string inner_indent = indent + INDENT;

    for (const auto& pair : get_string_members(inner_indent))
    {
      str << inner_indent << pair.first << ": " << pair.second << ",\n";
    }
    str << indent << "}";
    return str.str();
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const = 0;
};

class Path
{
public:
  Path(std::string original_path)
    : Path(std::filesystem::path(original_path))
  {
  }

  Path(std::filesystem::path original_path)
    : Path(original_path, {})
  {
  }

  Path(const Path& other)
    : m_original_path(other.m_original_path)
    , m_update_paths(other.m_update_paths)
  {
  }

  Path(std::filesystem::path original_path, std::vector<std::filesystem::path> update_paths)
    : m_original_path(original_path)
  {
    for (std::filesystem::path update_path : update_paths)
    {
      if (std::filesystem::exists(update_path))
      {
        m_update_paths.push_back(update_path);
      }
    }
  }

  Path operator/(std::string other) const
  {
    std::vector<std::filesystem::path> update_paths;
    for (std::filesystem::path update_path : m_update_paths)
    {
      update_paths.push_back(update_path / other);
    }
    return Path(m_original_path / other, update_paths);
  }

  bool exists() const
  {
    for (std::filesystem::path update_path : m_update_paths)
    {
      if (std::filesystem::exists(update_path))
      {
        return true;
      }
    }
    return std::filesystem::exists(m_original_path);
  }

  std::filesystem::path std() const
  {
    for (std::filesystem::path update_path : m_update_paths)
    {
      if (std::filesystem::exists(update_path))
      {
        return update_path;
      }
    }
    return m_original_path;
  }

  operator std::filesystem::path() const
  {
    return std();
  }

  std::string string() const
  {
    return std().string();
  }

  operator std::string() const
  {
    return string();
  }

  std::string filename() const
  {
    return std().filename();
  }

  std::vector<Path> list() const
  {
    std::set<std::string> child_names;
    std::vector<Path> result;

    std::vector<std::filesystem::path> ordered_paths = m_update_paths;
    ordered_paths.push_back(m_original_path);

    for (std::filesystem::path update_path : ordered_paths)
    {
      if (std::filesystem::exists(update_path))
      {
        for (const auto& entry : std::filesystem::directory_iterator(update_path))
        {
          std::string child_name = entry.path().filename();
          if (child_names.find(child_name) == child_names.end())
          {
            child_names.insert(child_name);
            result.push_back(*this / child_name);
          }
        }
      }
    }

    return result;
  }

private:
  std::filesystem::path m_original_path;
  std::vector<std::filesystem::path> m_update_paths;
};

class TimeSequence
{
public:
  struct IndexedTimestamp
  {
    template <typename TIterator, typename = decltype(std::declval<TIterator&&>()->first)>
    IndexedTimestamp(TIterator&& it)
      : timestamp(it->first)
      , index(it->second)
    {
    }

    IndexedTimestamp()
    {
    }

    uint64_t timestamp;
    size_t index;
  };

  TimeSequence(xt::xtensor<uint64_t, 1>&& timestamps, std::filesystem::path path)
    : m_timestamps(std::move(timestamps))
    , m_path(path)
  {
    size_t index = 0;
    for (uint64_t timestamp : m_timestamps)
    {
      m_timestamp_to_index[timestamp] = index;
      index++;
    }
  }

  TimeSequence()
    : m_timestamps({})
    , m_path("")
  {
  }

  const xt::xtensor<uint64_t, 1>& get_timestamps() const
  {
    return m_timestamps;
  }

  size_t get_length() const
  {
    return m_timestamps.shape()[0];
  }

  std::pair<IndexedTimestamp, IndexedTimestamp> get_nearest2(uint64_t timestamp) const
  {
    std::pair<IndexedTimestamp, IndexedTimestamp> result;
    auto it = m_timestamp_to_index.lower_bound(timestamp);
    if (it == m_timestamp_to_index.begin())
    {
      // Before first timestamp
      IndexedTimestamp first(m_timestamp_to_index.begin());
      result.first = first;
      result.second = first;
    }
    else if (it == m_timestamp_to_index.end())
    {
      // After last timestamp
      IndexedTimestamp last(m_timestamp_to_index.rbegin());
      result.first = last;
      result.second = last;
    }
    else
    {
      auto prev_it = it;
      prev_it--;
      result.first = IndexedTimestamp(prev_it);
      result.second = IndexedTimestamp(it);
    }

    if  (((result.first.index == result.second.index) != (result.first.timestamp == result.second.timestamp))
      || ((result.first.index <  result.second.index) != (result.first.timestamp <  result.second.timestamp))
      || result.first.index > result.second.index
      || result.first.timestamp > result.second.timestamp)
    {
      throw std::runtime_error("TimeSequence: Got indices " + std::to_string(result.first.index) + ", " + std::to_string(result.second.index) + " and timestamps "
         + std::to_string(result.first.timestamp) + " and " + std::to_string(result.second.timestamp) + " in " + m_path.string());
    }
    return result;
  }

  IndexedTimestamp get_nearest1(uint64_t timestamp) const
  {
    auto nearest2 = get_nearest2(timestamp);
    if (timestamp - nearest2.first.timestamp < nearest2.second.timestamp - timestamp)
    {
      return nearest2.first;
    }
    else
    {
      return nearest2.second;
    }
  }

private:
  xt::xtensor<uint64_t, 1> m_timestamps;
  std::filesystem::path m_path;
  std::map<uint64_t, size_t> m_timestamp_to_index;
};





class Data : public ToStringHelper
{
public:
  Data(uint64_t timestamp)
    : m_timestamp(timestamp)
  {
  }

  virtual ~Data() = default;

  uint64_t get_timestamp() const
  {
    return m_timestamp;
  }

  // virtual std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego) = 0;

  virtual void dummy() = 0;

  virtual py::tuple pickle() const = 0;

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result;
    result["timestamp"] = std::to_string(m_timestamp);
    return result;
  }

  static std::shared_ptr<Data> unpickle(py::tuple t);

private:
  uint64_t m_timestamp;
};

class Loader : public ToStringHelper
{
public:
  Loader(xt::xtensor<uint64_t, 1>&& timestamps, std::filesystem::path path)
    : m_timesequence(std::move(timestamps), path)
  {
  }

  Loader()
    : m_timesequence()
  {
  }

  Loader(TimeSequence&& timesequence)
    : m_timesequence(std::move(timesequence))
  {
  }

  virtual ~Loader() = default;

  virtual std::shared_ptr<Data> load(uint64_t timestamp, bool verbose) = 0;

  const TimeSequence& get_timesequence() const
  {
    return m_timesequence;
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result;
    if (this->get_timesequence().get_length() > 0)
    {
      result["timestamps"] = XTI_TO_STRING("np.ndarray(shape=(" << this->get_timesequence().get_length() << "))");
    }
    return result;
  }

private:
  TimeSequence m_timesequence;
};

class NamedData : public Data
{
public:
  NamedData(uint64_t timestamp, std::map<std::string, std::shared_ptr<Data>> data)
    : Data(timestamp)
    , m_data(data)
  {
  }

  virtual ~NamedData() = default;

  std::shared_ptr<Data> get(std::string name)
  {
    return m_data[name];
  }

  void remove(std::string name)
  {
    m_data.erase(name);
  }

  std::map<std::string, std::shared_ptr<Data>>& get_all()
  {
    return m_data;
  }

  const std::map<std::string, std::shared_ptr<Data>>& get_all() const
  {
    return m_data;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   std::map<std::string, std::shared_ptr<Data>> data;
  //   for (auto pair : m_data)
  //   {
  //     data[pair.first] = pair.second->move_ego(frame_loader, oldego_to_newego);
  //   }
  //   return std::make_shared<NamedData>(this->get_timestamp(), data);
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    for (const auto& data : m_data)
    {
      result[data.first] = data.second->to_string(inner_indent);
    }
    return result;
  }

  virtual py::tuple pickle() const
  {
    std::vector<py::object> result;
    result.push_back(py::cast(this->get_timestamp()));
    for (const auto& data : m_data)
    {
      result.push_back(py::make_tuple(py::cast(data.first), data.second->pickle()));
    }
    return py::make_tuple(py::cast("NamedData"), py::make_tuple(py::cast(result)));
  }

  static NamedData unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    std::vector<py::object> t0 = py::cast<std::vector<py::object>>(t[0]);
    std::map<std::string, std::shared_ptr<Data>> data;
    for (size_t i = 1; i < t0.size(); ++i)
    {
      auto pair = py::cast<std::pair<std::string, py::tuple>>(t0[i]);
      data[pair.first] = Data::unpickle(pair.second);
    }
    return NamedData(py::cast<uint64_t>(t0[0]), data);
  }

private:
  std::map<std::string, std::shared_ptr<Data>> m_data;
};

class NamedDataLoader : public Loader
{
public:
  NamedDataLoader(std::map<std::string, std::shared_ptr<Loader>> loaders)
    : m_loaders(loaders)
  {
  }
  virtual ~NamedDataLoader() = default;

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    std::map<std::string, std::shared_ptr<Data>> data;
    for (auto& loader : m_loaders)
    {
      data[loader.first] = loader.second->load(timestamp, verbose);
    }
    return std::make_shared<NamedData>(timestamp, data);
  }

  std::shared_ptr<Loader> get(std::string name)
  {
    return m_loaders[name];
  }

  void remove(std::string name)
  {
    m_loaders.erase(name);
  }

  std::map<std::string, std::shared_ptr<Loader>>& get_loaders()
  {
    return m_loaders;
  }

  const std::map<std::string, std::shared_ptr<Loader>>& get_loaders() const
  {
    return m_loaders;
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Loader::get_string_members(inner_indent);
    for (const auto& data : m_loaders)
    {
      result[data.first] = data.second->to_string(inner_indent);
    }
    return result;
  }

private:
  std::map<std::string, std::shared_ptr<Loader>> m_loaders;
};





class EgoToWorld : public Data  // xyz -> forward left up
{
public:
  EgoToWorld(uint64_t timestamp, cosy::Rigid<float, 3> transform)
    : Data(timestamp)
    , m_transform(transform)
  {
  }

  cosy::Rigid<float, 3> get_transform() const
  {
    return m_transform;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<EgoToWorld>(this->get_timestamp(), m_transform * oldego_to_newego.inverse());
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["transform"] = "cosy.Rigid";
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("EgoToWorld"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_transform)));
  }

  static EgoToWorld unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return EgoToWorld(py::cast<uint64_t>(t[0]), py::cast<cosy::Rigid<float, 3>>(t[1]));
  }

private:
  cosy::Rigid<float, 3> m_transform;
};

class EgoToWorldLoader : public Loader
{
public:
  static std::shared_ptr<EgoToWorldLoader> construct(Path path, bool verbose)
  {
    auto npz = load_npz(path / "ego_to_world.npz", verbose);
    xt::xtensor<uint64_t, 1> timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
    xt::xtensor<float, 3> matrices = load_from_npz_float<float, 3>(npz["transforms"]);
    return std::make_shared<EgoToWorldLoader>(std::move(timestamps), matrices, path);
  }

  EgoToWorldLoader(xt::xtensor<uint64_t, 1>&& timestamps, const xt::xtensor<float, 3>& matrices, std::filesystem::path path)
    : Loader(std::move(timestamps), path)
  {
    m_transformations.reserve(matrices.shape()[0]);
    for (size_t row = 0; row < matrices.shape()[0]; row++)
    {
      xti::mat4f m;
      for (size_t r = 0; r < 4; r++)
      {
        for (size_t c = 0; c < 4; c++)
        {
          m(r, c) = matrices(row, r, c);
        }
      }
      m_transformations.push_back(cosy::Rigid<float, 3>(m));
    }
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    return load2(timestamp, verbose);
  }

  std::shared_ptr<EgoToWorld> load2(uint64_t timestamp, bool verbose)
  {
    auto nearest = this->get_timesequence().get_nearest2(timestamp);
    cosy::Rigid<float, 3> transform;
    if (nearest.first.index == nearest.second.index)
    {
      transform = m_transformations[nearest.first.index];
    }
    else
    {
      float alpha = static_cast<float>(timestamp - nearest.first.timestamp) / static_cast<float>(nearest.second.timestamp - nearest.first.timestamp);
      if (alpha < 0 || alpha > 1)
      {
        throw std::runtime_error("Got invalid results from get_nearest2");
      }
      transform = cosy::slerp(m_transformations[nearest.first.index], m_transformations[nearest.second.index], alpha);
    }
    return std::make_shared<EgoToWorld>(timestamp, transform);
  }

private:
  std::vector<cosy::Rigid<float, 3>> m_transformations;
};










class GeoPose : public Data
{
public:
  GeoPose(uint64_t timestamp, xti::vec2d latlon, float bearing)
    : Data(timestamp)
    , m_latlon(latlon)
    , m_bearing(bearing)
  {
  }

  xti::vec2d get_latlon() const
  {
    return m_latlon;
  }

  float get_bearing() const
  {
    return m_bearing;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego_3d)
  // {
  //   // Check if ego_to_world is in frame_loader
  //   if (frame_loader->get_loaders().count("ego_to_world") == 0)
  //   {
  //     throw std::runtime_error("Scene does not contain ego_to_world data");
  //   }
  //   std::shared_ptr<EgoToWorldLoader> ego_to_world_loader = std::dynamic_pointer_cast<EgoToWorldLoader>(frame_loader->get_loaders().at("ego_to_world"));
  //   cosy::Rigid<float, 3> oldego_to_world_3d = ego_to_world_loader->load2(this->get_timestamp())->get_transform();
  //   auto _3d_to_2d = [](cosy::Rigid<float, 3> x_to_world){
  //     xti::vec3d forward = xt::linalg::dot(x_to_world.get_rotation(), xti::vec3d({1.0, 0.0, 0.0}));
  //     return cosy::Rigid(
  //       cosy::angle_to_rotation_matrix(cosy::angle_between_vectors(xti::vec2d({1.0, 0.0}), xti::vec2d({forward(0), forward(1)}))),
  //       xti::vec2d({x_to_world.get_translation()(0), x_to_world.get_translation()(1)})
  //     );
  //   };

  //   cosy::Rigid<float, 2> oldego_to_world_2d = _3d_to_2d(oldego_to_world_3d);
  //   cosy::Rigid<float, 2> newego_to_world_2d = _3d_to_2d(oldego_to_world_3d * oldego_to_newego_3d.inverse());

  //   cosy::proj::Transformer epsg3857_to_epsg4326("epsg:3857", "epsg:4326");
  //   cosy::proj::Transformer epsg4326_to_epsg3857 = *epsg3857_to_epsg4326.inverse();

  //   cosy::ScaledRigid<float, 2> oldego_to_epsg3857_2d = geopose_to_epsg3857(m_latlon, m_bearing, epsg4326_to_epsg3857);
  //   cosy::ScaledRigid<float, 2> world_to_epsg3857_2d = oldego_to_epsg3857_2d * cosy::ScaledRigid<float, 2>(oldego_to_world_2d.inverse());


  //   xti::vec2d latlon = epsg3857_to_epsg4326.transform(world_to_epsg3857_2d.transform(newego_to_world_2d.get_translation()));
  //   float bearing = cosy::degrees(epsg3857_to_epsg4326.transform_angle(cosy::rotation_matrix_to_angle((world_to_epsg3857_2d * cosy::ScaledRigid<float, 2>(newego_to_world_2d)).get_rotation())));

  //   return std::make_shared<GeoPose>(this->get_timestamp(), latlon, bearing);
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["latlon"] = XTI_TO_STRING("(" << m_latlon(0) << ", " << m_latlon(1) << ")");
    result["bearing"] = XTI_TO_STRING(m_bearing);
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("GeoPose"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_latlon), py::cast(m_bearing)));
  }

  static GeoPose unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return GeoPose(py::cast<uint64_t>(t[0]), py::cast<xti::vec2d>(t[1]), py::cast<float>(t[2]));
  }

private:
  xti::vec2d m_latlon;
  float m_bearing;
};

class GeoPoseLoader : public Loader
{
public:
  static std::shared_ptr<GeoPoseLoader> construct(Path path, bool verbose)
  {
    path = path / "geopose.npz";
    auto npz = load_npz(path, verbose);
    xt::xtensor<uint64_t, 1> timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
    xt::xtensor<float, 2> latlons = load_from_npz_float<float, 2>(npz["latlons"]);
    xt::xtensor<float, 1> bearings = load_from_npz_float<float, 1>(npz["bearings"]);
    if (timestamps.shape()[0] != latlons.shape()[0] || timestamps.shape()[0] != bearings.shape()[0] || latlons.shape()[1] != 2)
    {
      throw std::runtime_error(XTI_TO_STRING("Got invalid shapes in file " << path.string() << "\ntimestamps.shape=[" << timestamps.shape()[0] << "] latlons.shape=[" << latlons.shape()[0] << ", " << latlons.shape()[1] << "] bearings.shape=[" << bearings.shape()[0] << "]"));
    }
    return std::make_shared<GeoPoseLoader>(std::move(timestamps), std::move(latlons), std::move(bearings), path);
  }

  GeoPoseLoader(xt::xtensor<uint64_t, 1>&& timestamps, xt::xtensor<float, 2>&& latlons, xt::xtensor<float, 1>&& bearings, std::filesystem::path path)
    : Loader(std::move(timestamps), path)
    , m_latlons(std::move(latlons))
    , m_bearings(std::move(bearings))
  {
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    auto nearest = this->get_timesequence().get_nearest2(timestamp);
    xti::vec2d latlon;
    float bearing;
    if (nearest.first.index == nearest.second.index)
    {
      latlon = xt::view(m_latlons, nearest.first.index, xt::all());
      bearing = m_bearings(nearest.first.index);
    }
    else
    {
      float alpha = static_cast<float>(timestamp - nearest.first.timestamp) / static_cast<float>(nearest.second.timestamp - nearest.first.timestamp);
      if (alpha < 0 || alpha > 1)
      {
        throw std::runtime_error("Got invalid results from get_nearest2");
      }
      latlon = (1 - alpha) * xt::view(m_latlons, nearest.first.index, xt::all()) + alpha * xt::view(m_latlons, nearest.second.index, xt::all());
      bearing = (1 - alpha) * m_bearings(nearest.first.index) + alpha * m_bearings(nearest.second.index);
    }
    return std::make_shared<GeoPose>(timestamp, latlon, bearing);
  }

private:
  xt::xtensor<float, 2> m_latlons;
  xt::xtensor<float, 1> m_bearings;
};





class OutlierScore : public Data
{
public:
  OutlierScore(uint64_t timestamp, float score)
    : Data(timestamp)
    , m_score(score)
  {
  }

  float get_score() const
  {
    return m_score;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<OutlierScore>(*this);
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["score"] = XTI_TO_STRING(m_score);
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("OutlierScore"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_score)));
  }

  static OutlierScore unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return OutlierScore(py::cast<uint64_t>(t[0]), py::cast<float>(t[1]));
  }

private:
  float m_score;
};

class OutlierScoreLoader : public Loader
{
public:
  static std::shared_ptr<OutlierScoreLoader> construct(Path path, bool verbose)
  {
    path = path / "outlier_scores.npz";
    auto npz = load_npz(path, verbose);
    xt::xtensor<uint64_t, 1> timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
    xt::xtensor<float, 1> scores = load_from_npz_float<float, 1>(npz["scores"]);
    return std::make_shared<OutlierScoreLoader>(std::move(timestamps), std::move(scores), path);
  }

  OutlierScoreLoader(xt::xtensor<uint64_t, 1>&& timestamps, xt::xtensor<float, 1>&& scores, std::filesystem::path path)
    : Loader(std::move(timestamps), path)
    , m_scores(std::move(scores))
  {
  }

  const xt::xtensor<float, 1>& get_scores() const
  {
    return m_scores;
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    auto nearest = this->get_timesequence().get_nearest2(timestamp);
    float score;
    if (nearest.first.index == nearest.second.index)
    {
      score = m_scores(nearest.first.index);
    }
    else
    {
      float alpha = static_cast<float>(timestamp - nearest.first.timestamp) / static_cast<float>(nearest.second.timestamp - nearest.first.timestamp);
      if (alpha < 0 || alpha > 1)
      {
        throw std::runtime_error("Got invalid results from get_nearest2");
      }
      score = (1 - alpha) * m_scores(nearest.first.index) + alpha * m_scores(nearest.second.index);
    }
    return std::make_shared<OutlierScore>(timestamp, score);
  }

private:
  xt::xtensor<float, 1> m_scores;
};





class Camera : public Data // xyz -> right down forward
{
public:
  Camera(uint64_t timestamp, std::string name, xt::xtensor<uint8_t, 3>&& image, cosy::Rigid<float, 3> cam_to_ego, xti::mat3f projection)
    : Data(timestamp)
    , m_name(name)
    , m_image(std::move(image))
    , m_cam_to_ego(cam_to_ego)
    , m_projection(projection)
  {
  }

  std::string get_name() const
  {
    return m_name;
  }

  const xt::xtensor<uint8_t, 3>& get_image() const
  {
    return m_image;
  }

  void set_image(const xt::xtensor<uint8_t, 3>& image)
  {
    m_image = image;
  }

  cosy::Rigid<float, 3> get_cam_to_ego() const
  {
    return m_cam_to_ego;
  }

  xti::mat3f get_projection() const
  {
    return m_projection;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<Camera>(this->get_timestamp(), m_name, xt::xtensor<uint8_t, 3>(m_image), oldego_to_newego * m_cam_to_ego, m_projection);
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["name"] = "\"" + m_name + "\"";
    result["cam_to_ego"] = "cosy.Rigid";
    result["intr"] = "np.ndarray";
    result["image"] = XTI_TO_STRING("np.ndarray(shape=(" << m_image.shape()[0] << ", " << m_image.shape()[1] << ", " << m_image.shape()[2] << "))");
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("Camera"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_name), py::cast(m_image), py::cast(m_cam_to_ego), py::cast(m_projection)));
  }

  static Camera unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return Camera(py::cast<uint64_t>(t[0]), py::cast<std::string>(t[1]), py::cast<xt::xtensor<uint8_t, 3>>(t[2]), py::cast<cosy::Rigid<float, 3>>(t[3]), py::cast<xti::mat3f>(t[4]));
  }

private:
  std::string m_name;
  xt::xtensor<uint8_t, 3> m_image;
  cosy::Rigid<float, 3> m_cam_to_ego;
  xti::mat3f m_projection;
};

using CamToEgoMapper = std::function<cosy::Rigid<float, 3>(cosy::Rigid<float, 3> oldcam_to_ego, std::optional<cosy::Rigid<float, 3>> egotc_to_world)>;

class CameraLoader : public Loader
{
public:
  CameraLoader(std::filesystem::path path, std::string camera_name, std::string filetype, std::vector<cosy::Rigid<float, 3>>&& oldcam_to_ego, std::shared_ptr<EgoToWorldLoader> ego_to_world_loader, xt::xtensor<uint64_t, 1>&& timestamps, xti::vec2u resolution, xti::mat3f old_intr, xti::mat3f new_intr, std::vector<CamToEgoMapper> cam_to_ego_mappers = std::vector<CamToEgoMapper>())
    : CameraLoader(path, camera_name, filetype, std::move(oldcam_to_ego), ego_to_world_loader, TimeSequence(std::move(timestamps), path / "timestamps.npz"), resolution, old_intr, new_intr, std::move(cam_to_ego_mappers))
  {
  }

  CameraLoader(std::filesystem::path path, std::string camera_name, std::string filetype, std::vector<cosy::Rigid<float, 3>>&& oldcam_to_ego, std::shared_ptr<EgoToWorldLoader> ego_to_world_loader, TimeSequence&& timestamps, xti::vec2u resolution, xti::mat3f old_intr, xti::mat3f new_intr, std::vector<CamToEgoMapper> cam_to_ego_mappers = std::vector<CamToEgoMapper>())
    : Loader(std::move(timestamps))
    , m_path(path)
    , m_camera_name(camera_name)
    , m_filetype(filetype)
    , m_oldcam_to_ego(std::move(oldcam_to_ego))
    , m_ego_to_world_loader(ego_to_world_loader)
    , m_resolution(resolution)
    , m_old_intr(old_intr)
    , m_new_intr(new_intr)
    , m_cam_to_ego_mappers(cam_to_ego_mappers)
  {
  }

  static std::shared_ptr<CameraLoader> from_path(Path path, std::string camera_name, std::shared_ptr<EgoToWorldLoader> ego_to_world_loader, bool verbose)
  {
    Yaml metadata(path / "config.yaml", verbose);
    xt::xtensor<uint64_t, 1> timestamps;
    std::vector<cosy::Rigid<float, 3>> cam_to_ego;
    if (!std::filesystem::exists(path / "cam_to_ego.npz"))
    {
      auto npz = load_npz(path / "timestamps.npz", verbose);
      timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
      cam_to_ego.push_back(yaml_to_transform(metadata["cam_to_ego"]));
    }
    else
    {
      auto npz = load_npz(path / "cam_to_ego.npz", verbose);
      timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
      xt::xtensor<float, 3> matrices = load_from_npz_float<float, 3>(npz["transforms"]);
      cam_to_ego.reserve(matrices.shape()[0]);
      for (size_t row = 0; row < matrices.shape()[0]; row++)
      {
        xti::mat4f m;
        for (size_t r = 0; r < 4; r++)
        {
          for (size_t c = 0; c < 4; c++)
          {
            m(r, c) = matrices(row, r, c);
          }
        }
        cam_to_ego.push_back(cosy::Rigid<float, 3>(m));
      }
    }

    xti::mat3f intr = yaml_to_projection(metadata["intr"]);
    std::string filetype = metadata["filetype"].as<std::string>();
    xti::vec2u resolution({metadata["resolution"][0].as<uint32_t>(), metadata["resolution"][1].as<uint32_t>()});
    return std::make_shared<CameraLoader>(path, camera_name, filetype, std::move(cam_to_ego), ego_to_world_loader, std::move(timestamps), resolution, intr, intr);
  }

private:
  std::shared_ptr<CameraLoader> update(std::string camera_name, xti::vec2u resolution, xti::mat3f intr, std::vector<CamToEgoMapper> cam_to_ego_mappers)
  {
    return std::make_shared<CameraLoader>(m_path, camera_name, m_filetype, std::vector<cosy::Rigid<float, 3>>(m_oldcam_to_ego), m_ego_to_world_loader, TimeSequence(this->get_timesequence()), resolution, m_old_intr, intr, cam_to_ego_mappers);
  }

public:
  std::shared_ptr<CameraLoader> update(std::string camera_name, xti::vec2u resolution, xti::mat3f intr, CamToEgoMapper cam_to_ego_mapper)
  {
    std::vector<CamToEgoMapper> cam_to_ego_mappers = m_cam_to_ego_mappers;
    cam_to_ego_mappers.push_back(cam_to_ego_mapper);
    return update(camera_name, resolution, intr, cam_to_ego_mappers);
  }

  std::shared_ptr<CameraLoader> update(std::string camera_name, xti::vec2u resolution, xti::mat3f intr)
  {
    return update(camera_name, resolution, intr, m_cam_to_ego_mappers);
  }

  std::string get_name() const
  {
    return m_camera_name;
  }

  xti::vec2u get_resolution() const
  {
    return m_resolution;
  }

  xti::mat3f get_intr() const
  {
    return m_new_intr;
  }

  const std::vector<cosy::Rigid<float, 3>>& get_oldcam_to_ego() const
  {
    return m_oldcam_to_ego;
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    uint64_t cam_timestamp = this->get_timesequence().get_nearest1(timestamp).timestamp;

    // Load oldcam_to_ego, ego_to_world
    std::optional<cosy::Rigid<float, 3>> ego_to_world;
    cosy::Rigid<float, 3> oldcam_to_ego;
    if (m_ego_to_world_loader)
    {
      cosy::Rigid<float, 3> egotr_to_world = m_ego_to_world_loader->load2(timestamp, verbose)->get_transform();
      cosy::Rigid<float, 3> egotc_to_world = m_ego_to_world_loader->load2(cam_timestamp, verbose)->get_transform();

      ego_to_world = egotr_to_world;

      cosy::Rigid<float, 3> oldcam_to_egotc;
      if (m_oldcam_to_ego.size() == 1)
      {
        oldcam_to_egotc = m_oldcam_to_ego[0];
      }
      else
      {
        auto nearest = this->get_timesequence().get_nearest2(timestamp);
        if (nearest.first.index == nearest.second.index)
        {
          oldcam_to_egotc = m_oldcam_to_ego[nearest.first.index];
        }
        else
        {
          float alpha = static_cast<float>(timestamp - nearest.first.timestamp) / static_cast<float>(nearest.second.timestamp - nearest.first.timestamp);
          if (alpha < 0 || alpha > 1)
          {
            throw std::runtime_error("Got invalid results from get_nearest2");
          }
          oldcam_to_egotc = cosy::slerp(m_oldcam_to_ego[nearest.first.index], m_oldcam_to_ego[nearest.second.index], alpha);
        }
      }
      oldcam_to_ego = egotr_to_world.inverse() * egotc_to_world * oldcam_to_egotc;
    }
    else
    {
      if (timestamp != cam_timestamp)
      {
        throw std::runtime_error("Cannot interpolate timestamp without ego_to_world");
      }

      if (m_oldcam_to_ego.size() == 1)
      {
        oldcam_to_ego = m_oldcam_to_ego[0];
      }
      else
      {
        oldcam_to_ego = m_oldcam_to_ego[timestamp];
      }
    }

    // Update newcam_to_ego
    cosy::Rigid<float, 3> newcam_to_ego = oldcam_to_ego;
    for (const auto& cam_to_ego_mapper : m_cam_to_ego_mappers)
    {
      newcam_to_ego = cam_to_ego_mapper(newcam_to_ego, ego_to_world);
    }

    // Load image
    std::filesystem::path image_file = m_path / "images" / (std::to_string(cam_timestamp) + "." + m_filetype);
    cv::Mat image_cv = imread(image_file, verbose);

    // Anti-alias image
    float scale = m_new_intr(0, 0) / m_old_intr(0, 0);
    if (scale < 1)
    {
      float sigma = (1.0 / scale - 1) / 2;
      size_t kernel_size = static_cast<size_t>(std::ceil(sigma) * 4) + 1;
      cv::GaussianBlur(image_cv, image_cv, cv::Size(kernel_size, kernel_size), sigma, sigma);
    }

    // Compute homography
    cosy::Rigid<float, 3> ego_to_oldcam = oldcam_to_ego.inverse();
    cosy::Rigid<float, 3> ego_to_newcam = newcam_to_ego.inverse();

    cosy::Rigid<float, 3> oldcam_to_newcam = ego_to_newcam * ego_to_oldcam.inverse();

    xti::vec3f normal = xt::linalg::dot(ego_to_oldcam.get_rotation(), xti::vec3f({0, 0, 1}));
    float d_inv = 1.0 / xt::linalg::dot(normal, ego_to_oldcam.get_translation())();

    xt::xtensor<float, 2> homography_euclidean = oldcam_to_newcam.get_rotation() + d_inv * xt::linalg::outer(oldcam_to_newcam.get_translation(), normal);
    xt::xtensor<float, 2> homography = xt::linalg::dot(m_new_intr, xt::linalg::dot(homography_euclidean, xt::linalg::inv(m_old_intr)));
    homography /= homography(2, 2);
    homography_euclidean /= homography_euclidean(2, 2);

    // Remap image
    cv::Mat new_image_cv;
    cv::warpPerspective(image_cv, new_image_cv, xti::to_opencv(homography), cv::Size(m_resolution(1), m_resolution(0)), cv::INTER_LINEAR);
    image_cv = std::move(new_image_cv);

    // Image to tensor, bgr->rgb
    auto image_bgr = xt::view(xti::from_opencv<uint8_t>(std::move(image_cv)), xt::all(), xt::all(), xt::range(0, 3));
    xt::xtensor<uint8_t, 3> image_rgb = xt::view(std::move(image_bgr), xt::all(), xt::all(), xt::range(xt::placeholders::_, xt::placeholders::_, -1));

    return std::make_shared<Camera>(timestamp, m_camera_name, std::move(image_rgb), newcam_to_ego, m_new_intr);
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Loader::get_string_members(inner_indent);
    result["name"] = "\"" + m_camera_name + "\"";
    result["intr"] = "np.ndarray";
    result["resolution"] = XTI_TO_STRING("(" << m_resolution(0) << ", " << m_resolution(1) << ")");
    return result;
  }

private:
  std::filesystem::path m_path;
  std::string m_camera_name;
  std::string m_filetype;
  std::shared_ptr<EgoToWorldLoader> m_ego_to_world_loader;

  std::vector<cosy::Rigid<float, 3>> m_oldcam_to_ego;
  xti::mat3f m_old_intr;
  xti::mat3f m_new_intr;
  xti::vec2u m_resolution;
  std::vector<CamToEgoMapper> m_cam_to_ego_mappers;
};

namespace cam_ops {

class Op
{
public:
  virtual std::vector<std::shared_ptr<CameraLoader>> create(std::shared_ptr<CameraLoader> camera) const = 0;
};

class Tile : public Op
{
public:
  Tile(xti::vec2u tile_shape, std::optional<xti::vec2u> tile_crop_margin)
    : m_tile_shape(tile_shape)
  {
    if (tile_crop_margin)
    {
      m_tile_crop_margin = *tile_crop_margin;
    }
    else
    {
      m_tile_crop_margin = xti::vec2u({0, 0});
    }
  }

  std::vector<std::shared_ptr<CameraLoader>> create(std::shared_ptr<CameraLoader> camera) const
  {
    std::vector<std::shared_ptr<CameraLoader>> result;

    xti::vec2u resolution = camera->get_resolution();
    auto add_cam = [&](xti::vec2u offset, std::string camera_name){
      xti::vec2i new_resolution = resolution - offset;
      if (xt::any(m_tile_shape > new_resolution))
      {
        throw std::runtime_error(XTI_TO_STRING("tile_shape = " << m_tile_shape << " > " << new_resolution << " resolution"));
      }
      new_resolution = xt::minimum(m_tile_shape, new_resolution);

      xti::matXT<float, 3> new_intr = camera->get_intr();
      for (size_t r = 0; r < 2; r++)
      {
        new_intr(r, 2) -= offset(1 - r);
      }

      result.push_back(camera->update(camera_name, new_resolution, new_intr));
    };
    
    if (xt::all(resolution <= m_tile_shape + m_tile_crop_margin))
    {
      if (xt::any(resolution < m_tile_shape))
      {
        throw std::runtime_error("resolution < tile_shape");
      }
      add_cam((resolution - m_tile_shape) / 2, camera->get_name());
    }
    else
    {
      xti::vec2u tile_nums = (resolution + m_tile_shape - 1) / m_tile_shape;
      if (xt::any(tile_nums > 10))
      {
        throw std::runtime_error("Only supports tile index < 10");
      }
      xti::vec2u last_offset = resolution - m_tile_shape;
      for (uint32_t r = 0; r < tile_nums(0); r++)
      {
        for (uint32_t c = 0; c < tile_nums(1); c++)
        {
          xti::vec2u tile_index({r, c});
          xti::vec2u offset = last_offset * xt::cast<float>(tile_index) / xt::maximum(xt::cast<float>(tile_nums - 1), 1.0);
          std::string camera_name_tile = XTI_TO_STRING(camera->get_name() << "_t" << r << c);
          add_cam(offset, camera_name_tile);
        }
      }
    }

    return result;
  }

private:
  xti::vec2u m_tile_shape;
  xti::vec2u m_tile_crop_margin;
};

class Resize : public Op
{
public:
  Resize(std::function<float(const CameraLoader&)> scale_op)
    : m_scale_op(scale_op)
  {
  }

  std::vector<std::shared_ptr<CameraLoader>> create(std::shared_ptr<CameraLoader> camera) const
  {
    std::vector<std::shared_ptr<CameraLoader>> result;

    float scale = m_scale_op(*camera);

    xti::vec2u new_resolution = xt::cast<int32_t>(camera->get_resolution() * scale);
    xti::matXT<float, 3> new_intr = camera->get_intr();
    for (size_t r = 0; r < 2; r++)
    {
      for (size_t c = 0; c < 3; c++)
      {
        new_intr(r, c) *= scale;
      }
    }

    result.push_back(camera->update(camera->get_name(), new_resolution, new_intr));

    return result;
  }

private:
  std::function<float(const CameraLoader&)> m_scale_op;
};

class Filter : public Op
{
public:
  Filter(std::function<bool(const CameraLoader&)> filter)
    : m_filter(filter)
  {
  }

  std::vector<std::shared_ptr<CameraLoader>> create(std::shared_ptr<CameraLoader> camera) const
  {
    std::vector<std::shared_ptr<CameraLoader>> result;

    if (m_filter(*camera))
    {
      result.push_back(camera);
    }

    return result;
  }

private:
  std::function<bool(const CameraLoader&)> m_filter;
};

class Homography : public Op
{
public:
  Homography(CamToEgoMapper mapper)
    : m_mapper(mapper)
  {
  }

  std::vector<std::shared_ptr<CameraLoader>> create(std::shared_ptr<CameraLoader> camera) const
  {
    std::vector<std::shared_ptr<CameraLoader>> result;

    result.push_back(camera->update(camera->get_name(), camera->get_resolution(), camera->get_intr(), m_mapper));

    return result;
  }

private:
  CamToEgoMapper m_mapper;
};

} // end of ns cam_ops




class Lidar : public Data
{
public:
  Lidar(uint64_t timestamp, std::string name, xt::xtensor<float, 2>&& points)
    : Data(timestamp)
    , m_name(name)
    , m_points(std::move(points))
  {
  }

  std::string get_name() const
  {
    return m_name;
  }

  const xt::xtensor<float, 2>& get_points() const
  {
    return m_points;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<Lidar>(this->get_timestamp(), m_name, oldego_to_newego.transform_all(m_points));
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["name"] = "\"" + m_name + "\"";
    result["points"] = XTI_TO_STRING("np.ndarray(shape=(" << m_points.shape()[0] << ", " << m_points.shape()[1] << "))");
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("Lidar"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_name), py::cast(m_points)));
  }

  static Lidar unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return Lidar(py::cast<uint64_t>(t[0]), py::cast<std::string>(t[1]), py::cast<xt::xtensor<float, 2>>(t[2]));
  }

private:
  std::string m_name;
  xt::xtensor<float, 2> m_points;
};

class LidarLoader : public Loader
{
public:
  static std::shared_ptr<LidarLoader> construct(Path path, std::shared_ptr<EgoToWorldLoader> ego_to_world_loader, bool verbose)
  {
    auto npz = load_npz(path / "timestamps.npz", verbose);
    xt::xtensor<uint64_t, 1> timestamps = load_from_npz_int<uint64_t, 1>(npz["timestamps"]);
    return std::make_shared<LidarLoader>(path, std::move(timestamps), ego_to_world_loader);
  }

  LidarLoader(std::filesystem::path path, xt::xtensor<uint64_t, 1>&& timestamps, std::shared_ptr<EgoToWorldLoader> ego_to_world_loader)
    : Loader(std::move(timestamps), path / "timestamps.npz")
    , m_path(path)
    , m_ego_to_world_loader(ego_to_world_loader)
  {
  }

  std::string get_name() const
  {
    return m_path.filename();
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    uint64_t lidar_timestamp = this->get_timesequence().get_nearest1(timestamp).timestamp;

    std::filesystem::path points_file = m_path / "points" / (std::to_string(lidar_timestamp) + ".npz");
    auto npz = load_npz(points_file, verbose);
    std::string key = "";
    if (npz.size() == 1)
    {
      key = npz.begin()->first;
    }
    else if (npz.count("arr_0"))
    {
      key = "arr_0";
    }
    else if (npz.count("points"))
    {
      key = "points";
    }
    else
    {
      throw std::runtime_error(XTI_TO_STRING("Npz file has more than one key"));
    }

    xt::xtensor<float, 2> points = load_from_npz_float<float, 2>(npz[key]);

    if (m_ego_to_world_loader)
    {
      cosy::Rigid<float, 3> egotr_to_world = m_ego_to_world_loader->load2(timestamp, verbose)->get_transform();
      cosy::Rigid<float, 3> egotl_to_world = m_ego_to_world_loader->load2(lidar_timestamp, verbose)->get_transform();
      cosy::Rigid<float, 3> egotl_to_egotr = egotr_to_world.inverse() * egotl_to_world;
      points = egotl_to_egotr.transform_all(std::move(points));
    }
    else
    {
      if (timestamp != lidar_timestamp)
      {
        throw std::runtime_error("Cannot interpolate timestamp without ego_to_world");
      }
    }

    return std::make_shared<Lidar>(timestamp, this->get_name(), std::move(points));
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Loader::get_string_members(inner_indent);
    result["name"] = get_name();
    return result;
  }

private:
  std::filesystem::path m_path;
  std::shared_ptr<EgoToWorldLoader> m_ego_to_world_loader;
};

namespace lidar_ops {

class Op
{
public:
  virtual std::vector<std::shared_ptr<LidarLoader>> create(std::shared_ptr<LidarLoader> camera) const = 0;
};

class Filter : public Op
{
public:
  Filter(std::function<bool(const LidarLoader&)> filter)
    : m_filter(filter)
  {
  }

  std::vector<std::shared_ptr<LidarLoader>> create(std::shared_ptr<LidarLoader> camera) const
  {
    std::vector<std::shared_ptr<LidarLoader>> result;

    if (m_filter(*camera))
    {
      result.push_back(camera);
    }

    return result;
  }

private:
  std::function<bool(const LidarLoader&)> m_filter;
};

} // end of ns lidar_ops




class Lidars : public NamedData
{
public:
  Lidars(uint64_t timestamp, std::map<std::string, std::shared_ptr<Data>> data)
    : NamedData(timestamp, data)
  {
  }

  Lidars(NamedData other)
    : NamedData(other)
  {
  }

  xt::xtensor<float, 2> get_points()
  {
    xt::xtensor<float, 2> points = xt::xtensor<float, 2>({0, 3});
    for (auto pair : this->get_all())
    {
      points = xt::concatenate(xt::xtuple(std::move(points), std::static_pointer_cast<Lidar>(pair.second)->get_points()), 0);
    }
    return points;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<Lidars>(static_cast<NamedData&>(*this->NamedData::move_ego(frame_loader, oldego_to_newego)));
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = NamedData::get_string_members(inner_indent);
    uint32_t total_num_points = 0;
    for (auto pair : this->get_all())
    {
      total_num_points += std::static_pointer_cast<Lidar>(pair.second)->get_points().shape()[0];
    }
    result["points"] = XTI_TO_STRING("np.ndarray(shape=(" << total_num_points << ", " << 3 << "))");
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("Lidars"), this->NamedData::pickle());
  }

  static Lidars unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return Lidars(NamedData::unpickle(t));
  }
};

class LidarsLoader : public NamedDataLoader
{
public:
  LidarsLoader(std::map<std::string, std::shared_ptr<Loader>> loaders)
    : NamedDataLoader(loaders)
  {
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    auto super_loaded = std::static_pointer_cast<NamedData>(this->NamedDataLoader::load(timestamp, verbose));
    return std::make_shared<Lidars>(timestamp, super_loaded->get_all());
  }
};





class Map : public Data
{
public:
  Map(uint64_t timestamp, std::string name, xt::xtensor<uint8_t, 3>&& image, float meters_per_pixel)
    : Data(timestamp)
    , m_name(name)
    , m_image(std::move(image))
    , m_meters_per_pixel(meters_per_pixel)
  {
  }

  std::string get_name() const
  {
    return m_name;
  }

  const xt::xtensor<uint8_t, 3>& get_image() const
  {
    return m_image;
  }

  void set_image(const xt::xtensor<uint8_t, 3>& image)
  {
    m_image = image;
  }

  float get_meters_per_pixel() const
  {
    return m_meters_per_pixel;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   throw std::runtime_error("move_ego is not implemented for Map");
  // }

  void dummy() {}

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Data::get_string_members(inner_indent);
    result["name"] = "\"" + m_name + "\"";
    result["meters_per_pixel"] = std::to_string(m_meters_per_pixel);
    result["image"] = XTI_TO_STRING("np.ndarray(shape=(" << m_image.shape()[0] << ", " << m_image.shape()[1] << ", " << m_image.shape()[2] << "))");
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("Map"), py::make_tuple(py::cast(this->get_timestamp()), py::cast(m_name), py::cast(m_image), py::cast(m_meters_per_pixel)));
  }

  static Map unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return Map(py::cast<uint64_t>(t[0]), py::cast<std::string>(t[1]), py::cast<xt::xtensor<uint8_t, 3>>(t[2]), py::cast<float>(t[3]));
  }

private:
  std::string m_name;
  xt::xtensor<uint8_t, 3> m_image;
  float m_meters_per_pixel;
};

class MapLoader : public Loader
{
public:
  MapLoader(std::filesystem::path path, std::string name, std::string filetype, xt::xtensor<float, 1>&& meters_per_pixel, xt::xtensor<uint64_t, 1>&& timestamps, xti::vec2u resolution)
    : Loader(std::move(timestamps), path / "timestamps.npz")
    , m_path(path)
    , m_name(name)
    , m_filetype(filetype)
    , m_meters_per_pixel(std::move(meters_per_pixel))
    , m_resolution(resolution)
  {
  }

  static std::shared_ptr<MapLoader> from_path(Path path, std::string name, bool verbose)
  {
    Yaml metadata(path / "config.yaml", verbose);
    auto timestamps_npz = load_npz(path / "timestamps.npz", verbose);
    xt::xtensor<uint64_t, 1> timestamps = load_from_npz_int<uint64_t, 1>(timestamps_npz["timestamps"]);

    xt::xtensor<float, 1> meters_per_pixel;
    if (std::filesystem::exists(path / "meters_per_pixel.npz"))
    {
      auto meters_per_pixel_npz = load_npz(path / "meters_per_pixel.npz", verbose);
      meters_per_pixel = load_from_npz_float<float, 1>(meters_per_pixel_npz["meters_per_pixel"]);
    }
    else
    {
      meters_per_pixel = xt::xtensor<float, 1>({1});
      meters_per_pixel(0) = metadata["meters_per_pixel"].as<float>();
    }

    std::string filetype = metadata["filetype"].as<std::string>();
    xti::vec2u resolution({metadata["resolution"][0].as<uint32_t>(), metadata["resolution"][1].as<uint32_t>()});
    return std::make_shared<MapLoader>(path, name, filetype, std::move(meters_per_pixel), std::move(timestamps), resolution);
  }

  std::string get_name() const
  {
    return m_name;
  }

  xti::vec2u get_resolution() const
  {
    return m_resolution;
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    auto nearest = this->get_timesequence().get_nearest1(timestamp);
    if (timestamp != nearest.timestamp)
    {
      throw std::runtime_error("Cannot interpolate timestamp for MapLoader");
    }

    // Load image
    std::filesystem::path image_file = m_path / "images" / (std::to_string(timestamp) + "." + m_filetype);
    cv::Mat image_cv = imread(image_file, verbose);
    auto image_bgr = xt::view(xti::from_opencv<uint8_t>(std::move(image_cv)), xt::all(), xt::all(), xt::range(0, 3));
    xt::xtensor<uint8_t, 3> image_rgb = xt::view(std::move(image_bgr), xt::all(), xt::all(), xt::range(xt::placeholders::_, xt::placeholders::_, -1));

    return std::make_shared<Map>(timestamp, m_name, std::move(image_rgb), m_meters_per_pixel(nearest.index));
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = Loader::get_string_members(inner_indent);
    result["name"] = "\"" + m_name + "\"";
    result["resolution"] = XTI_TO_STRING("(" << m_resolution(0) << ", " << m_resolution(1) << ")");
    return result;
  }

private:
  std::filesystem::path m_path;
  std::string m_name;
  std::string m_filetype;
  xt::xtensor<float, 1> m_meters_per_pixel;
  xti::vec2u m_resolution;
};







class Frame : public NamedData
{
public:
  Frame(std::string scene_name, std::string location, std::string dataset, std::shared_ptr<NamedData> data, std::string name)
    : NamedData(*data)
    , m_scene_name(scene_name)
    , m_location(location)
    , m_dataset(dataset)
    , m_name(name)
  {
  }

  Frame(std::string scene_name, std::string location, std::string dataset, NamedData&& data, std::string name)
    : NamedData(std::move(data))
    , m_scene_name(scene_name)
    , m_location(location)
    , m_dataset(dataset)
    , m_name(name)
  {
  }

  std::string get_scene_name() const
  {
    return m_scene_name;
  }

  std::string get_location() const
  {
    return m_location;
  }

  std::string get_dataset() const
  {
    return m_dataset;
  }

  std::string get_name() const
  {
    return m_name;
  }

  // std::shared_ptr<Data> move_ego(std::shared_ptr<FrameLoader> frame_loader, cosy::Rigid<float, 3> oldego_to_newego)
  // {
  //   return std::make_shared<Frame>(m_scene_name, m_location, m_dataset, std::static_pointer_cast<NamedData>(this->NamedData::move_ego(frame_loader, oldego_to_newego)), m_name);
  // }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = NamedData::get_string_members(inner_indent);
    result["scene_name"] = "\"" + m_scene_name + "\"";
    result["location"] = "\"" + m_location+ "\"";
    result["dataset"] = "\"" + m_dataset+ "\"";
    result["name"] = "\"" + m_name + "\"";
    return result;
  }

  virtual py::tuple pickle() const
  {
    return py::make_tuple(py::cast("Frame"), py::make_tuple(py::cast(m_scene_name), py::cast(m_location), py::cast(m_dataset), this->NamedData::pickle(), py::cast(m_name)));
  }

  static Frame unpickle(py::tuple t)
  {
    t = t[1].cast<py::tuple>();
    return Frame(py::cast<std::string>(t[0]), py::cast<std::string>(t[1]), py::cast<std::string>(t[2]), NamedData::unpickle(t[3]), py::cast<std::string>(t[4]));
  }

private:
  std::string m_scene_name;
  std::string m_location;
  std::string m_dataset;
  std::string m_name;
};

class FrameLoader : public NamedDataLoader
{
public:
  static std::shared_ptr<FrameLoader> construct(std::filesystem::path std_path, std::vector<std::shared_ptr<cam_ops::Op>> cam_ops, std::vector<std::shared_ptr<lidar_ops::Op>> lidar_ops, std::vector<std::filesystem::path> std_updates, bool verbose)
  {
    if (verbose)
    {
      std::cout << "cvgl_data: Loading scene from " << std_path << std::endl;
    }
    Path path(std_path, std_updates);

    std::filesystem::path yaml_path = (path / "config.yaml").std();
    if (!std::filesystem::exists(yaml_path))
    {
      throw std::runtime_error(XTI_TO_STRING("File " << yaml_path.string() << " is missing"));
    }
    Yaml metadata(yaml_path, verbose);

    std::string location = metadata["location"].as<std::string>();
    std::string dataset = metadata["dataset"].as<std::string>();

    std::map<std::string, std::shared_ptr<Loader>> loaders;

    Path ego_to_world_path = path / "ego_to_world.npz";
    std::shared_ptr<EgoToWorldLoader> ego_to_world_loader;
    if (std::filesystem::exists(ego_to_world_path))
    {
      ego_to_world_loader = EgoToWorldLoader::construct(path, verbose);
      loaders["ego_to_world"] = ego_to_world_loader;
    }

    Path camera_path = path / "camera";
    if (std::filesystem::exists(camera_path))
    {
      if (verbose)
      {
        std::cout << "cvgl_data: Loading cameras from " << camera_path.string() << std::endl;
      }
      std::map<std::string, std::shared_ptr<CameraLoader>> camera_loaders;
      for (auto child_path : camera_path.list())
      {
        if (std::filesystem::is_directory(child_path))
        {
          std::string camera_name = child_path.filename();
          if (camera_loaders.count(camera_name))
          {
            throw std::runtime_error(XTI_TO_STRING("Camera " << camera_name << " already exists"));
          }
          try
          {
            camera_loaders[camera_name] = CameraLoader::from_path(child_path, camera_name, ego_to_world_loader, verbose);
          }
          catch (const std::runtime_error& e)
          {
            throw std::runtime_error(XTI_TO_STRING("Error while loading camera meta-data at " << child_path.string() << ":\n" << e.what()));
          }
        }
      }
      for (const auto& op : cam_ops)
      {
        std::map<std::string, std::shared_ptr<CameraLoader>> new_camera_loaders;
        for (const auto& old_camera_pair : camera_loaders)
        {
          for (const auto& new_camera : op->create(old_camera_pair.second))
          {
            if (new_camera_loaders.count(new_camera->get_name()))
            {
              throw std::runtime_error(XTI_TO_STRING("Camera " << new_camera->get_name() << " already exists"));
            }
            new_camera_loaders[new_camera->get_name()] = new_camera;
          }
        }
        camera_loaders = std::move(new_camera_loaders);
      }

      std::map<std::string, std::shared_ptr<Loader>> camera_loaders2;
      for (const auto& [camera_name, camera] : camera_loaders)
      {
        camera_loaders2[camera_name] = camera;
      }
      loaders["camera"] = std::make_shared<NamedDataLoader>(camera_loaders2);
    }

    if (std::filesystem::exists(path / "geopose.npz"))
    {
      std::shared_ptr<GeoPoseLoader> geopose_loader = GeoPoseLoader::construct(path, verbose);
      loaders["geopose"] = geopose_loader;
    }

    if (std::filesystem::exists(path / "outlier_scores.npz"))
    {
      std::shared_ptr<OutlierScoreLoader> outlier_score_loader = OutlierScoreLoader::construct(path, verbose);
      loaders["outlier_score"] = outlier_score_loader;
    }

    Path lidar_path = path / "lidar";
    if (std::filesystem::exists(lidar_path))
    {
      if (verbose)
      {
        std::cout << "cvgl_data: Loading lidars from " << lidar_path.string() << std::endl;
      }
      std::map<std::string, std::shared_ptr<LidarLoader>> lidar_loaders;
      for (auto child_path : lidar_path.list())
      {
        if (std::filesystem::is_directory(child_path))
        {
          std::string lidar_name = child_path.filename();
          if (lidar_loaders.count(lidar_name))
          {
            throw std::runtime_error(XTI_TO_STRING("Lidar " << lidar_name << " already exists"));
          }
          try
          {
            lidar_loaders[lidar_name] = LidarLoader::construct(child_path, ego_to_world_loader, verbose);
          }
          catch (const std::runtime_error& e)
          {
            throw std::runtime_error(XTI_TO_STRING("Error while loading lidar meta-data at " << child_path.string() << ":\n" << e.what()));
          }
        }
      }
      for (const auto& op : lidar_ops)
      {
        std::map<std::string, std::shared_ptr<LidarLoader>> new_lidar_loaders;
        for (const auto& old_lidar_pair : lidar_loaders)
        {
          for (const auto& new_lidar : op->create(old_lidar_pair.second))
          {
            if (new_lidar_loaders.count(new_lidar->get_name()))
            {
              throw std::runtime_error(XTI_TO_STRING("Lidar " << new_lidar->get_name() << " already exists"));
            }
            new_lidar_loaders[new_lidar->get_name()] = new_lidar;
          }
        }
        lidar_loaders = std::move(new_lidar_loaders);
      }

      std::map<std::string, std::shared_ptr<Loader>> lidar_loaders2;
      for (const auto& [name, lidar] : lidar_loaders)
      {
        lidar_loaders2[name] = lidar;
      }
      loaders["lidar"] = std::make_shared<LidarsLoader>(lidar_loaders2);
    }
    else
    {
      std::map<std::string, std::shared_ptr<Loader>> lidar_loaders;
      loaders["lidar"] = std::make_shared<LidarsLoader>(lidar_loaders);
    }

    Path map_path = path / "map";
    if (std::filesystem::exists(map_path))
    {
      std::map<std::string, std::shared_ptr<Loader>> loaders;
      for (auto child_path : map_path.list())
      {
        if (std::filesystem::is_directory(child_path))
        {
          std::string map_name = child_path.filename();
          if (loaders.count(map_name))
          {
            throw std::runtime_error(XTI_TO_STRING("Map " << map_name << " already exists"));
          }
          loaders[map_name] = MapLoader::from_path(child_path, map_name, verbose);
        }
      }
      
      loaders["map"] = std::make_shared<NamedDataLoader>(loaders);
    }

    return std::make_shared<FrameLoader>(path, loaders, location, dataset);
  }

  FrameLoader(std::filesystem::path path, std::map<std::string, std::shared_ptr<Loader>> loaders, std::string location, std::string dataset)
    : NamedDataLoader(loaders)
    , m_path(path)
    , m_location(location)
    , m_dataset(dataset)
  {
  }

  std::string get_scene_name() const
  {
    return m_path.filename();
  }

  std::string get_location() const
  {
    return m_location;
  }

  std::string get_dataset() const
  {
    return m_dataset;
  }

  std::shared_ptr<Data> load(uint64_t timestamp, bool verbose)
  {
    std::string name = m_dataset + "-" + get_scene_name() + "-t" + std::to_string(timestamp);
    return std::make_shared<Frame>(this->get_scene_name(), m_location, m_dataset, std::static_pointer_cast<NamedData>(this->NamedDataLoader::load(timestamp, verbose)), name);
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result = NamedDataLoader::get_string_members(inner_indent);
    result["scene_name"] = "\"" + this->get_scene_name()+ "\"";
    result["location"] = "\"" + m_location+ "\"";
    result["dataset"] = "\"" + m_dataset+ "\"";
    return result;
  }

private:
  std::filesystem::path m_path;
  std::string m_location;
  std::string m_dataset;
};





class TiledWebMapsLoader : public ToStringHelper
{
public:
  TiledWebMapsLoader(std::shared_ptr<tiledwebmaps::TileLoader> tileloader, std::string name, size_t zoom)
    : m_tileloader(tileloader)
    , m_name(name)
    , m_zoom(zoom)
  {
  }

  std::string get_name() const
  {
    return m_name;
  }

  std::shared_ptr<tiledwebmaps::TileLoader> get_tileloader() const
  {
    return m_tileloader;
  }

  size_t get_zoom() const
  {
    return m_zoom;
  }

  std::shared_ptr<Data> load(xti::vec2f latlon, float bearing, float meters_per_pixel, xti::vec2s shape, std::string location, bool verbose)
  {
    std::map<std::string, std::shared_ptr<Data>> data;

    // Image is loaded with bearing pointing upwards in image, but we want it to point in positive first axis direction, so add 180deg to bearing
    data["map"] = std::make_shared<Map>(0, m_name, tiledwebmaps::load_metric(*m_tileloader, latlon, bearing + 180.0, meters_per_pixel, shape, m_zoom), meters_per_pixel);
    data["geopose"] = std::make_shared<GeoPose>(0, latlon, bearing);

    return std::make_shared<Frame>(m_name, location, m_name, std::make_shared<NamedData>(0, data), XTI_TO_STRING(m_name << "-z" << m_zoom << "-lat" << latlon(0) << "-lon" << latlon(1) << "-b" << bearing));
  }

  virtual std::map<std::string, std::string> get_string_members(std::string inner_indent) const
  {
    std::map<std::string, std::string> result;
    result["name"] = "\"" + m_name + "\"";
    result["zoom"] = std::to_string(m_zoom);
    result["tileloader"] = "tiledwebmaps.TileLoader";
    return result;
  }

private:
  std::shared_ptr<tiledwebmaps::TileLoader> m_tileloader;
  std::string m_name;
  size_t m_zoom;
};




std::shared_ptr<Data> Data::unpickle(py::tuple t)
{
  std::string classname = t[0].cast<std::string>();
  if (classname == "EgoToWorld")
  {
    return std::make_shared<EgoToWorld>(EgoToWorld::unpickle(t));
  }
  else if (classname == "GeoPose")
  {
    return std::make_shared<GeoPose>(GeoPose::unpickle(t));
  }
  else if (classname == "OutlierScore")
  {
    return std::make_shared<OutlierScore>(OutlierScore::unpickle(t));
  }
  else if (classname == "Camera")
  {
    return std::make_shared<Camera>(Camera::unpickle(t));
  }
  else if (classname == "Lidar")
  {
    return std::make_shared<Lidar>(Lidar::unpickle(t));
  }
  else if (classname == "Lidars")
  {
    return std::make_shared<Lidars>(Lidars::unpickle(t));
  }
  else if (classname == "Map")
  {
    return std::make_shared<Map>(Map::unpickle(t));
  }
  else if (classname == "Frame")
  {
    return std::make_shared<Frame>(Frame::unpickle(t));
  }
  else if (classname == "NamedData")
  {
    return std::make_shared<NamedData>(NamedData::unpickle(t));
  }
  else
  {
    throw std::runtime_error("Data: Unknown class " + classname);
  }
}

} // end of ns cvgl_data
