cmake_minimum_required(VERSION 3.10)
project(untitled)

set(CMAKE_CXX_STANDARD 17)  # C++17 사용

# OpenCV 설정 (정확한 경로)
set(OpenCV_DIR "C:/Users/ych97/opencv-mingw")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories("C:/Users/ych97/opencv-mingw/x64/mingw/lib")  # 명시적으로 라이브러리 경로 추가

add_executable(untitled main.cpp)

# OpenCV 라이브러리 자동 연결 (수동 추가 필요 없음)
target_link_libraries(untitled ${OpenCV_LIBS})
