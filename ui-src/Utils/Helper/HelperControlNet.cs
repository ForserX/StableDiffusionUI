using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    class ControlNetBase
    {
        protected string Model;
        protected string CNModel;

        public string GetModelPathSD()
        {   
            return FS.GetModelDir() + "controlnet/" + Model;
        }

        public string GetModelPathCN()
        {
            return FS.GetModelDir() + "controlnet/" + CNModel;
        }

        virtual public void CheckSD()
        {
        }

        virtual public void CheckCN()
        {
        }

        virtual public string CommandLine()
        {
            return "";
        }
    }

    class ControlNetOpenPose: ControlNetBase
    {
        public ControlNetOpenPose()
        {
            Model = "sd-controlnet-openpose";
            CNModel = "OpenposeDetector/anannotator/ckpts/body_pose_model.pth";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadCNPoser();
                Notification.SendNotification("Downloading pose model: done!");
            }
        }

        public override void CheckCN()
        {
            if (!System.IO.File.Exists(GetModelPathCN()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadSDCNPoser();
                Notification.SendNotification("Download pose model: done!");
            }
        }
        override public string CommandLine()
        {
            string cmdline = $" --pose=\"{Helper.CurrentPose}\"";

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\"";
            }
            else
            {
                cmdline += $" --mode=\"IfP\"";
            }

            return cmdline;
        }
    }

    class ControlNetHed: ControlNetBase
    {
        public ControlNetHed()
        {
            Model = "sd-controlnet-hed";
            CNModel = "OpenposeDetector/anannotator/ckpts/network-bsds500.pth";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadSDCNHed();
                Notification.SendNotification("Downloading pose model: done!");
            }
        }

        public override void CheckCN()
        {
            if (!System.IO.File.Exists(GetModelPathCN()))
            {
                Notification.SendNotification("Starting downloading hed model...");
                WGetDownloadModels.DownloadSDCNPoser();
                Notification.SendNotification("Download hed model: done!");
            }
        }
        override public string CommandLine()
        {
            string cmdline = $" --pose=\"{Helper.CurrentPose}\"";

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\"";
            }
            else
            {
                cmdline += $" --mode=\"IfP\"";
            }

            return cmdline;
        }
    }

    class ControlNetCanny : ControlNetBase
    {
        public ControlNetCanny()
        {
            Model = "sd-controlnet-canny";
            CNModel = "";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading canny model...");
                WGetDownloadModels.DownloadSDCNCanny();
                Notification.SendNotification("Downloading canny model: done!");
            }
        }

        override public string CommandLine()
        {
            string cmdline = $" --pose=\"{Helper.CurrentPose}\"";

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\" ";
            }
            else
            {
                cmdline += $" --mode=\"IfP\" ";
            }

            return cmdline;
        }
    }

    class ControlNetDepth: ControlNetBase
    {
        public ControlNetDepth()
        {
            Model = "sd-controlnet-depth";
            CNModel = "";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading depth model...");
                WGetDownloadModels.DownloadSDCNDepth();
                Notification.SendNotification("Downloading depth model: done!");
            }
        }

        override public string CommandLine()
        {
            string cmdline = $" --pose=\"{Helper.CurrentPose}\"";

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\" ";
            }
            else
            {
                cmdline += $" --mode=\"IfP\" ";
            }

            return cmdline;
        }
    }

    class ControlNetNormal : ControlNetBase
    {
        public ControlNetNormal()
        {
            Model = "sd-controlnet-normal";
            CNModel = "";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading normal model...");
                WGetDownloadModels.DownloadSDCNNormal();
                Notification.SendNotification("Downloading pose normal: done!");
            }
        }

        override public string CommandLine()
        {
            string cmdline = $" --pose=\"{Helper.CurrentPose}\"";

            if (Helper.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\" ";
            }
            else
            {
                cmdline += $" --mode=\"IfP\" ";
            }

            return cmdline;
        }
    }

    // Global Functions
    internal class HelperControlNet
    {
        public static ControlNetOpenPose OpenPose = new ControlNetOpenPose();
        public static ControlNetCanny Canny = new ControlNetCanny();
        public static ControlNetDepth Depth = new ControlNetDepth();
        public static ControlNetHed Hed = new ControlNetHed();
        public static ControlNetNormal Normal = new ControlNetNormal();

        public enum ControlTypes
        {
            Poser,
            Canny,
            Depth,
            Normal,
            Seg,
            Scribble,
            Hed,
            mlsd
        }
    }
}
