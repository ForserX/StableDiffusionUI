﻿using System.IO;
using System.Collections.Generic;
using System.Windows.Media.Imaging;
using OpenCvSharp;
using System.Windows.Media;
using OpenCvSharp.WpfExtensions;
using System.Drawing;

namespace SD_FXUI
{
    struct LoadData
    {
        public int FramePerSecond;
        public string File;
    }

    public struct VideoData
    {
        public bool FrameDone;
        public bool ActiveRender;
        public int ActiveFrame;
        public List<string> Files;
    }

    internal class VideoFrame
    {
        static LoadData LastVideoLoad;

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
        public static void ReadVideo(string Path, int FPS)
        {
            bool bCheck = LastVideoLoad.File == Path && LastVideoLoad.FramePerSecond == FPS;

            if (!bCheck)
            {
                Host.Print("Extract video captures...");

                FS.Dir.Delete(CapturePath, true);
                Directory.CreateDirectory(CapturePath);

                GlobalVariables.LastVideoData.FrameDone = false;
                GlobalVariables.LastVideoData.ActiveRender = true;
                GlobalVariables.LastVideoData.ActiveFrame = 0;
                GlobalVariables.LastVideoData.Files = new List<string>();

                VideoCapture Capture = new VideoCapture(Path);
                var ImgMat = new Mat();

                int DeltaFPS = 0;
                if (FPS < Capture.Fps)
                {
                    double fDeltaFPS = (double)FPS / Capture.Fps;
                    fDeltaFPS *= 10;
                    DeltaFPS = (int)fDeltaFPS;
                }
                else
                {
                    DeltaFPS = 999999999;
                }

                uint FrameIt = 0;

                while (Capture.IsOpened())
                {
                    Capture.Read(ImgMat);

                    if (ImgMat.Empty())
                        break;

                    if (FrameIt % DeltaFPS != 0)
                    {
                        SaveFrames(OpenCvSharp.WpfExtensions.BitmapSourceConverter.ToBitmapSource(ImgMat), FrameIt);
                    }

                    FrameIt++;

                    if (Cv2.WaitKey(1) == 113)
                        break;
                }

                LastVideoLoad.File = Path;
                LastVideoLoad.FramePerSecond = FPS;
            }
        }
        public static void RenderVideo(string Input)
        {
            Host.Print("Render video...");
            VideoCapture InCapture = new VideoCapture($"{Input}%d.png");

            //#TODO: Upscaler coef 
            //OpenCvSharp.Size OutVideoSize = new OpenCvSharp.Size();
            //OutVideoSize.Width = GlobalVariables.MakeInfo.Width * GlobalVariables.MakeInfo.sc

            VideoWriter OutCapture = new VideoWriter($"{GlobalVariables.ImgPath}test.mp4", FourCC.AVC, 25, new OpenCvSharp.Size(GlobalVariables.MakeInfo.Width, GlobalVariables.MakeInfo.Height));

            Mat FrameImage = new Mat();
            while (true)
            {
                InCapture.Read(FrameImage);
                if (FrameImage.Empty())
                    break;

                OutCapture.Write(FrameImage);
            }

            Host.Print("Render video... Done!");
            Notification.SendNotification("Video render: Done!");
        }

        internal static ImageSource GetPreviewPic()
        {
            VideoCapture Capture = new VideoCapture(GlobalVariables.InputImagePath);
            Mat FirstFrameCapture = new Mat();

            Capture.Read(FirstFrameCapture);
            Capture.Release();

            Bitmap OutputPriview = CodeUtils.BitmapFromSource(FirstFrameCapture.ToBitmapSource());
            Bitmap PlayIcon = CodeUtils.BitmapFromSource(GlobalVariables.Form.IconPlay.Source);

            PlayIcon = CodeUtils.ResizeBitmap(PlayIcon, new System.Drawing.Size(OutputPriview.Height, OutputPriview.Height));

            int SizeDeltaX = OutputPriview.Width - PlayIcon.Width;
            SizeDeltaX /= 4;

            Graphics g = Graphics.FromImage(OutputPriview);
            g.DrawImage(PlayIcon, new System.Drawing.Point(SizeDeltaX, 0));

            return CodeUtils.CreateBitmapSourceFromGdiBitmap(OutputPriview);
        }
    }
}
