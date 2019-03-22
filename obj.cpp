#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
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
    // List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[6] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

    #if (CV_MINOR_VERSION < 3)
    {
        tracker = Tracker::create(trackerType);
    }
    #else
    {
        if (trackerType == "BOOSTING")
            tracker = TrackerBoosting::create();
        if (trackerType == "MIL")
            tracker = TrackerMIL::create();
        if (trackerType == "KCF")
            tracker = TrackerKCF::create();
        if (trackerType == "TLD")
            tracker = TrackerTLD::create();
        if (trackerType == "MEDIANFLOW")
            tracker = TrackerMedianFlow::create();
        if (trackerType == "GOTURN")
            tracker = TrackerGOTURN::create();
    }
    #endif
    // Read video
    int minx = 994/2, miny = 0;
    int width = 2*minx, height = 1080;
    double speed = 4.;
    Size2i output_sz{width, height};

    VideoCapture video(argv[1]);
    VideoWriter writer(argv[2], VideoWriter::fourcc('H','2','6','4'), video.get(CAP_PROP_FPS), Size{width, height});

    // Exit if video is not opened
    if(!video.isOpened())
    {
        cerr << "Could not read video file\n";
        return 1;
    }

    // Exit if video is not opened
    if(!writer.isOpened())
    {
        cerr << "Could not write video file\n";
        return 4;
    }

    // Read first frame
    video.set(CV_CAP_PROP_POS_MSEC, 15000);
    Mat frame;
    /* bool ok = video.read(frame); */
    /* if (!ok) { */
    /*     cout << "Could not read frame\n"; */
    /*     return 2; */
    /* } */

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
    while(video.read(frame))
    {
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);

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
        double max_shift = 0;
        if (!init) {
            init = true;
            tl = Point2i{(int)std::round(newx),(int)std::round(newy)};
        } else {
            max_shift = std::max(newx-tl.x, newy-tl.y);
            tl.x += (int)std::trunc((newx - tl.x)/speed);
            tl.y += (int)std::trunc((newy - tl.y)/speed);
        }

        if (tl.x < 0) tl.x = 0;
        if (tl.x > frame.cols - width) tl.x = frame.cols - width;
        if (tl.y < 0) tl.y = 0;
        if (tl.y > frame.rows - height) tl.y = frame.rows - height;
        Rect2i bbox2{tl, output_sz};
        Mat result{frame, bbox2};
        writer.write(result);
        rectangle(frame, bbox2, Scalar( 0, 255, 0 ), 2, 1 );

        // Tracking success : Draw the tracked object
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );

        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)) + ", shift " + SSTR(max_shift), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        float sec = video.get(CAP_PROP_POS_MSEC) / 1000;
        int framePos = video.get(CAP_PROP_POS_FRAMES);
        putText(frame, "Pos : " + SSTR(sec) + "s " + SSTR(framePos), Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
        if (framePos % 16 == 0)
            imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
    }
}
