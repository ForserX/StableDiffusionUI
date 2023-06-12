using System.IO;
using System.Collections.Generic;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using Windows.Devices.Geolocation;

namespace SD_FXUI
{
    public struct VideoData
    {
        public bool FrameDone;
        public bool ActiveRender;
        public int ActiveFrame;
        public List<string> Files;
    }

    internal class VideoFrame
    {
        static public string CapturePath = FS.GetImagesDir() + "vidcap\\";
        static void SaveFrames(BitmapSource Src, uint It)
        {
            BitmapSource image = Src;
            string FilePath = $"{CapturePath}{It}.png";
            GlobalVariables.LastVideoData.Files.Add(FilePath);

            using (var fileStream = new FileStream(FilePath, FileMode.Create))
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                //encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Save(fileStream);
            }
        }
        public static void ReadVideo(string Path)
        {
            FS.Dir.Delete(CapturePath, true);
            Directory.CreateDirectory(CapturePath);

            GlobalVariables.LastVideoData.FrameDone = false;
            GlobalVariables.LastVideoData.ActiveRender = false;
            GlobalVariables.LastVideoData.ActiveFrame = 0;
            GlobalVariables.LastVideoData.Files = new List<string>();

            VideoCapture Capture = new VideoCapture(Path);
            var image = new Mat();

            uint FrameIt = 0;

            while (Capture.IsOpened())
            {
                Capture.Read(image);

                if (image.Empty())
                    break;

                SaveFrames(OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(image), FrameIt);

                FrameIt++;

                if (Cv2.WaitKey(1) == 113)
                    break;
            }
            //RenderVideo();
        }
        public static void RenderVideo(string Input)
        {
            VideoCapture InCapture = new VideoCapture($"{Input}%d.png");
            //VideoCapture OutCapture = new VideoCapture("N:\\TestMP4\\test.mp4", VideoCaptureAPIs.FFMPEG);

            OpenCvSharp.VideoWriter OutCapture = new VideoWriter($"{GlobalVariables.ImgPath}test.mp4", FourCC.AVC, 25, new OpenCvSharp.Size(GlobalVariables.MakeInfo.Width, GlobalVariables.MakeInfo.Height));

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
