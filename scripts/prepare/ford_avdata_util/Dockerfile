FROM osrf/ros:kinetic-desktop-full

SHELL ["/bin/bash", "-c"]

WORKDIR /

RUN apt-get update -y && apt-get install -y libpcap-dev

RUN source /opt/ros/kinetic/setup.bash && mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone https://github.com/fferflo/velodyne.git -b add-queue-size && \
    git clone https://github.com/fferflo/perception_pcl.git -b add-queue-size && \
    cd /catkin_ws && catkin_make

COPY . /ford-avdata-code

RUN /ford-avdata-code/entrypoint.sh

ENTRYPOINT ["/ford-avdata-code/entrypoint.sh"]
CMD ["bash"]
