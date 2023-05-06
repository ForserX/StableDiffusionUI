using ABI.System;
using SD_FXUI.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class WGetDownloadModels
    {
        public static void DownloadCNPoser(HelperControlNet.ControlTypes Type)
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet/";

            Directory.CreateDirectory(WorkingDir);
            Directory.CreateDirectory(WorkingDir + "anannotator");
            Directory.CreateDirectory(WorkingDir + "anannotator\\ckpts");

            if (Type == HelperControlNet.ControlTypes.Poser)
            {
                FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth", WorkingDir + @"anannotator/ckpts/body_pose_model.pth");
                FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth", WorkingDir + @"anannotator/ckpts/hand_pose_model.pth");
                FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth", WorkingDir + @"anannotator/ckpts/facenet.pth");
            }

            if (Type == HelperControlNet.ControlTypes.Hed)
            {
                FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth", WorkingDir + @"anannotator/ckpts/network-bsds500.pth");
            }

            if (Type == HelperControlNet.ControlTypes.Mlsd)
            {
                FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth", WorkingDir + @"anannotator/ckpts/mlsd_large_512_fp32.pth");
            }
        }
        public static void DownloadSDPoser()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-openpose/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/ControlNetMediaPipeFace-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.fp16.bin", WorkingDir + @"diffusion_pytorch_model.bin");
        }
        
        public static void DownloadSDFacegen()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-facegen/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/ControlNetMediaPipeFace-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
            FileDownloader.DownloadFileAsync("https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/raw/main/diffusion_sd15/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/blob/main/diffusion_sd15/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
        }

        public static void DownloadSDCNCanny()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-canny/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-canny/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-canny-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }

        public static void DownloadSDCNDepth()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-depth/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-depth/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-depth-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }

        public static void DownloadSDCNHed()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-hed/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-hed/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-hed-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }

        public static void DownloadSDCNNormal()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-normal/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-normal/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-normal-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }

        public static void DownloadSDCNScribble()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-scribble/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-scribble/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-scribble-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }
        public static void DownloadSDCNSeg()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-seg/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-seg/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-seg-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }
        public static void DownloadSDCNMLSD()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-mlsd/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-mlsd/raw/main/config.json", WorkingDir + @"config.json");
            FileDownloader.DownloadFileAsync("https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/diffusion_pytorch_model.bin", WorkingDir + @"diffusion_pytorch_model.bin");
            FileDownloader.DownloadFileAsync("https://huggingface.co/ForserX/sd-controlnet-mlsd-onnx/resolve/main/model.onnx", WorkingDir + @"model.onnx");
        }
    }
}
