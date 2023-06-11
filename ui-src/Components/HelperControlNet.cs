using MetadataExtractor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Automation;

namespace SD_FXUI
{
    class ControlNetBase
    {
        protected HelperControlNet.ControlTypes ControlType;
        protected string Model = "";
        protected string CNModel = "";
        protected string BaseOutPath = FS.GetModelDir() + "controlnet/gen/";

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
            string cmdline = $" --pose=\"{GlobalVariables.CurrentPose}\"";

            if (GlobalVariables.Mode == Helper.ImplementMode.ONNX)
            {
                cmdline += $" --mode=\"IfPONNX\"";
            }
            else
            {
                cmdline += $" --mode=\"IfP\"";
            }

            return cmdline;
        }

        virtual public string PreprocessCommandLine()
        {
            return "";
        }

        virtual public string Outdir()
        {
            return BaseOutPath;
        }

        public string GetModelName()
        {
            return Model;
        }
    }

    class ControlNetOpenPose: ControlNetBase
    {
        public ControlNetOpenPose()
        {
            Model = "sd-controlnet-openpose";
            CNModel = "sd-controlnet/anannotator/ckpts/";
            ControlType = HelperControlNet.ControlTypes.OpenPose;
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadSDPoser();
                Notification.SendNotification("Downloading pose model: done!");
            }
        }

        public override void CheckCN()
        {
            if (!System.IO.File.Exists(GetModelPathCN() + "hand_pose_model.pth"))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadCNPoser(ControlType);
                Notification.SendNotification("Download pose model: done!");
            }
        }

        override public string PreprocessCommandLine()
        {
            return "PfI";
        }

        override public string Outdir()
        {
            return FS.GetModelDir(FS.ModelDirs.OpenPose);
        }
    }

    class ControlNetFace: ControlNetBase
    {
        public ControlNetFace()
        {
            Model = "sd-controlnet-facegen";
            CNModel = "";

            ControlType = HelperControlNet.ControlTypes.MediapipeFace;
            System.IO.Directory.CreateDirectory(Outdir());
        }

        public override void CheckSD()
        {
            if (!System.IO.Directory.Exists(GetModelPathSD()))
            {
                Notification.SendNotification("Starting downloading pose model...");
                WGetDownloadModels.DownloadSDFacegen();
                Notification.SendNotification("Downloading pose model: done!");
            }
        }

        override public string PreprocessCommandLine()
        {
            return "FfI";
        }

        override public string Outdir()
        {
            return FS.GetModelDir() + "OpenFaces/";
        }
    }

    class ControlNetHed: ControlNetBase
    {
        public ControlNetHed()
        {
            Model = "sd-controlnet-hed";
            CNModel = "sd-controlnet/anannotator/ckpts/network-bsds500.pth";
            ControlType = HelperControlNet.ControlTypes.Hed;

            System.IO.Directory.CreateDirectory(Outdir());
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
                WGetDownloadModels.DownloadCNPoser(ControlType);
                Notification.SendNotification("Download hed model: done!");
            }
        }
       
        override public string PreprocessCommandLine()
        {
            return "HfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "Hed/";
        }
    }

    class ControlNetCanny : ControlNetBase
    {
        public ControlNetCanny()
        {
            Model = "sd-controlnet-canny";
            CNModel = "";
            ControlType = HelperControlNet.ControlTypes.Canny;

            System.IO.Directory.CreateDirectory(Outdir());
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

        override public string PreprocessCommandLine()
        {
            return "CfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "Canny/";
        }
    }

    class ControlNetDepth: ControlNetBase
    {
        public ControlNetDepth()
        {
            Model = "sd-controlnet-depth";
            CNModel = "";
            ControlType = HelperControlNet.ControlTypes.Depth;

            System.IO.Directory.CreateDirectory(Outdir());
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

        override public string PreprocessCommandLine()
        {
            return "DfI";
        }

        override public string Outdir()
        {
            return BaseOutPath + "Depth/";
        }
    }

    class ControlNetNormal : ControlNetBase
    {
        public ControlNetNormal()
        {
            Model = "sd-controlnet-normal";
            CNModel = "";
            ControlType = HelperControlNet.ControlTypes.NormalMap;

            System.IO.Directory.CreateDirectory(Outdir());
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

        override public string PreprocessCommandLine()
        {
            return "NfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "Normal/";
        }
    }

    class ControlNetScribble : ControlNetHed
    {
        public ControlNetScribble()
        {
            Model = "sd-controlnet-scribble";
            CNModel = "sd-controlnet/anannotator/ckpts/network-bsds500.pth";
            ControlType = HelperControlNet.ControlTypes.Scribble;

            System.IO.Directory.CreateDirectory(Outdir());
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

        override public string PreprocessCommandLine()
        {
            return "SfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "Scribble/";
        }
    }

    class ControlNetSeg : ControlNetBase
    {
        public ControlNetSeg()
        {
            Model = "sd-controlnet-seg";
            CNModel = "";
            ControlType = HelperControlNet.ControlTypes.Segmentation;

            System.IO.Directory.CreateDirectory(Outdir());
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

        override public string PreprocessCommandLine()
        {
            return "SgfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "Seg/";
        }
    }

    class ControlNetMLSD : ControlNetBase
    {
        public ControlNetMLSD()
        {
            Model = "sd-controlnet-mlsd";
            CNModel = "sd-controlnet\\anannotator\\ckpts\\mlsd_large_512_fp32.pth";
            ControlType = HelperControlNet.ControlTypes.Mlsd;

            System.IO.Directory.CreateDirectory(Outdir());
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
                WGetDownloadModels.DownloadCNPoser(ControlType);
                Notification.SendNotification("Download mlsd model: done!");
            }
        }

        override public string PreprocessCommandLine()
        {
            return "MfI";
        }
        override public string Outdir()
        {
            return BaseOutPath + "mlsd/";
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
        public static ControlNetFace Face = new ControlNetFace();

        public static ControlNetBase Current = null;
        public enum ControlTypes
        {
            Poser,
            Canny,
            Depth,
            Depth_leres,
            Hed,
            NormalMap,
            OpenPose,
            MediapipeFace,
            Clip_vision,
            Color,
            Pidinet,
            Segmentation,
            Mlsd,
            Scribble,
            Fake_Scribble,
            Binary
        }

        public static ControlNetBase GetType(string StrData)
        {
            string LoverName = StrData.ToLower();

            if (LoverName == "canny")               return Canny;
            if (LoverName == "depth")               return Depth;
            if (LoverName == "hed")                 return Hed;
            if (LoverName == "normalmap")           return Normal;
            if (LoverName == "openposedetector")    return OpenPose;
            if (LoverName == "scribble")            return Scribble;
            if (LoverName == "segmentation")        return Seg;
            if (LoverName == "mlsd")                return MLSD;
            if (LoverName == "facegen")             return Face;

            /*
			if (LoverName == "depth_leres") ; // not implemented return canny
			if (LoverName == "openpose_hand") ; // not implemented return canny
			if (LoverName == "clip_vision") ; // not implemented return canny
			if (LoverName == "fake_scribble") ; // not implemented return canny
			if (LoverName == "pidinet") ; // not implemented return canny
			if (LoverName == "binary")         ; // not implemented return canny
			*/

            return HelperControlNet.Canny;                      // temp Bypass error;

        }
    }
}
