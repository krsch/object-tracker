#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace cv::cudacodec;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
{
    if (argc <= 2) {
        fprintf(stderr, "Usage: stabilize <input> <output>\n");
        return -1;
    }
    Ptr<Tracker> tracker;

    // Read video
    int minx = 784/2, miny = 40;
    int width = 2*minx, height = 851;
    Size2i output_sz{width, height};

    auto video = createVideoReader(std::string(argv[1]));
    double fps;
    {
        VideoCapture video("test.mkv");
        fps = video.get(CAP_PROP_FPS);
    }
    auto writer = createVideoWriter(std::string(argv[2]), Size{width, height}, fps);

    // Read first frame
    /* video.set(CV_CAP_PROP_POS_MSEC, 15000); */
    Mat frame;
    // Define initial boundibg box
    Rect2d bbox(384, 327, 465-384, 440-327);

    // Uncomment the line below to select a different bounding box
    /* bbox = selectROI("Tracking", frame, false); */

    // Display bounding box.
    /* rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); */
    /* imshow("Tracking", frame); */
    /* tracker->init(frame, bbox); */
    // Start timer
    auto timer = getTickCount();
    Point2i tl;
    bool init = false;
    int frames = 0;
    while(video->nextFrame(frame))
    {
        // Update the tracking result
        bool ok = init && tracker->update(frame, bbox);

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / (getTickCount() - timer);
        timer = getTickCount();

        if (!ok) {
            bbox = selectROI("Tracking", frame, false);
            cout << bbox << endl;
            if (bbox.tl() == bbox.br())
                return 0;
            tracker = TrackerKCF::create();
            if (! tracker->init(frame, bbox))
                cerr << "Couldn't initialize tracker\n";
        }

        double newx = (bbox.tl().x + bbox.br().x) / 2 - minx;
        double newy = bbox.tl().y - miny;
        if (!init) tl = Point2i{(int)std::round(newx),(int)std::round(newy)};
        else {
            tl.x += (int)std::trunc((newx - tl.x)/16.);
            tl.y += (int)std::trunc((newy - tl.y)/16.);
        }

        if (tl.x < 0) tl.x = 0;
        if (tl.x > frame.cols - width) tl.x = frame.cols - width;
        if (tl.y < 0) tl.y = 0;
        if (tl.y > frame.rows - height) tl.y = frame.rows - height;
        Rect2i bbox2{tl, output_sz};
        Mat result{frame, bbox2};
        writer->write(result);
        rectangle(frame, bbox2, Scalar( 0, 255, 0 ), 2, 1 );

        // Tracking success : Draw the tracked object
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );

        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        float sec = frames / fps;
        putText(frame, "Pos : " + SSTR(sec) + "s " + SSTR(frames), Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        ++frames;

        // Display frame.
        if (frames % 16 == 0)
            imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
    }
}
