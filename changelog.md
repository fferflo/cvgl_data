# Change Log

## [0.1.3] - 2023-07-21
 
### Added

- Added `verbose=True` option to `cvgl_data.load` and `scene.load`.
- Added option to remove sensors, e.g. `del scene["lidar"]` and `del frame["camera"]`.

### Changed

- Improved error handling and reporting.

### Fixed

- Fixed bug that caused "Illegal instruction (core dumped)" on some devices. (Remove -march=native)
- Fixed homography computation for cameras with same origin.