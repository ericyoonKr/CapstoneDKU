#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat img_frame;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Unable to open camera" << endl;
        return -1;
    }

    // To get image size, capture one image
    if (!cap.read(img_frame)) {
        cout << "Unable to read camera" << endl;
        return -1;
    }

    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); // 코덱 설정
    double fps = 30.0;
    VideoWriter writer("output.avi", codec, fps, img_frame.size());

    if (!writer.isOpened()) {
        cout << "Unable to open output file" << endl;
        return -1;
    }

    while (true) {
        // Capture image
        if (!cap.read(img_frame)) {
            cout << "Unable to read camera" << endl;
            break;
        }

        writer.write(img_frame); // 비디오 파일에 프레임 저장
        imshow("output", img_frame); // 화면에 출력

        int key = waitKey(1);
        if (key == 27) { // ESC 키를 누르면 종료
            break;
        }
    }

    // 루프 종료 후 리소스 해제
    cap.release();
    writer.release();
    destroyAllWindows();

    return 0;
}
