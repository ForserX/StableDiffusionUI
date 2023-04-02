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

        virtual public string PreprocessCommandLine()
        {
            return "";
        }
    }

    class ControlNetOpenPose: ControlNetBase
    {
        public ControlNetOpenPose()
        {
            Model = "sd-controlnet-openpose";
            CNModel = "sd-controlnet/anannotator/ckpts/body_pose_model.pth";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadCNPoser(HelperControlNet.ControlTypes.Poser);
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

        override public string PreprocessCommandLine()
        {
            return "PfI";
        }
    }

    class ControlNetHed: ControlNetBase
    {
        public ControlNetHed()
        {
            Model = "sd-controlnet-hed";
            CNModel = "sd-controlnet/anannotator/ckpts/network-bsds500.pth";
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
                WGetDownloadModels.DownloadCNPoser(HelperControlNet.ControlTypes.Hed);
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

        override public string PreprocessCommandLine()
        {
            return "HfI";
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

        override public string PreprocessCommandLine()
        {
            return "CfI";
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

        override public string PreprocessCommandLine()
        {
            return "DfI";
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

        override public string PreprocessCommandLine()
        {
            return "NfI";
        }
    }

    class ControlNetScribble : ControlNetBase
    {
        public ControlNetScribble()
        {
            Model = "sd-controlnet-scribble";
            CNModel = "";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading scribble model...");
                WGetDownloadModels.DownloadSDCNScribble();
                Notification.SendNotification("Downloading pose scribble: done!");
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

        override public string PreprocessCommandLine()
        {
            return "SfI";
        }
    }

    class ControlNetSeg : ControlNetBase
    {
        public ControlNetSeg()
        {
            Model = "sd-controlnet-seg";
            CNModel = "";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading seg model...");
                WGetDownloadModels.DownloadSDCNSeg();
                Notification.SendNotification("Downloading seg model: done!");
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

        override public string PreprocessCommandLine()
        {
            return "SgfI";
        }
    }

    class ControlNetMLSD : ControlNetBase
    {
        public ControlNetMLSD()
        {
            Model = "sd-controlnet-mlsd";
            CNModel = "sd-controlnet/annotator/ckpts/mlsd_large_512_fp32.pth";
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading mlsd model...");
                WGetDownloadModels.DownloadSDCNMLSD();
                Notification.SendNotification("Downloading mlsd model: done!");
            }
        }

        public override void CheckCN()
        {
            if (!System.IO.File.Exists(GetModelPathCN()))
            {
                Notification.SendNotification("Starting downloading mlsd model...");
                WGetDownloadModels.DownloadCNPoser(HelperControlNet.ControlTypes.Mlsd);
                Notification.SendNotification("Download mlsd model: done!");
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

        override public string PreprocessCommandLine()
        {
            return "MfI";
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
        public static ControlNetScribble Scribble = new ControlNetScribble();
        public static ControlNetSeg Seg = new ControlNetSeg();
        public static ControlNetMLSD MLSD = new ControlNetMLSD();

        public enum ControlTypes
        {
            Poser,
            Canny,
            Depth,
            Depth_leres,
            Hed,
            NormalMap,
            OpenPose,
            OpenPose_hand,
            Clip_vision,
            Color,
            Pidinet,
            Segmentation,
            Mlsd,
            Scribble,
            Fake_Scribble,
            Binary
        }
    }
}
