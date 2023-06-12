using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using OpenCvSharp;

namespace SD_FXUI
{
    internal class VideoFrame
    {
        static void SaveFrames(BitmapSource Src, uint It)
        {
            BitmapSource image = Src;
            using (var fileStream = new FileStream($"N:\\TestMP4\\{It}.png", FileMode.Create))
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                //encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Save(fileStream);
            }
        }
        public static void ReadVideo(string Path)
        {
            VideoCapture Capture = new VideoCapture(Path);
            var image = new Mat();

            uint FrameRate = (uint)Capture.FrameCount / (uint)Capture.Fps;
            FrameRate /= 2;

            uint FrameIt = 0;
            uint TotalIt = 0;

            while (Capture.IsOpened())
            {
                Capture.Read(image);

                if (image.Empty())
                    break;
                
                if (FrameIt == TotalIt)
                {
                    SaveFrames(OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(image), TotalIt / FrameRate);
                    TotalIt += FrameRate;
                }

                FrameIt++;

                if (Cv2.WaitKey(1) == 113)
                    break;
            }
            RenderVideo();
        }
        static void RenderVideo()
        {
            VideoCapture InCapture = new VideoCapture("N:\\TestMP4\\%d.png");
            //VideoCapture OutCapture = new VideoCapture("N:\\TestMP4\\test.mp4", VideoCaptureAPIs.FFMPEG);

            OpenCvSharp.VideoWriter OutCapture = new VideoWriter("N:\\TestMP4\\test.mp4", FourCC.AVC, 25, new OpenCvSharp.Size(600, 400));

            Mat FrameImage = new Mat();
            while (true)
            {
                InCapture.Read(FrameImage);
                if (FrameImage.Empty())
                    break;

                OutCapture.Write(FrameImage);
            }
        }
    }
}
