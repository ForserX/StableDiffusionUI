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
        public static void DownloadUpscalerONNX()
        {
            string WorkingDir = FS.GetModelDir() + "upscaler/onnx/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }

            Directory.CreateDirectory(WorkingDir);
            Directory.CreateDirectory(WorkingDir + "low_res_scheduler");
            Directory.CreateDirectory(WorkingDir + "scheduler");
            Directory.CreateDirectory(WorkingDir + "text_encoder");
            Directory.CreateDirectory(WorkingDir + "tokenizer");
            Directory.CreateDirectory(WorkingDir + "unet");
            Directory.CreateDirectory(WorkingDir + "vae");

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();
            Download.Send(WGetFile + "-O \"low_res_scheduler/scheduler_config.json\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/low_res_scheduler/scheduler_config.json");

            Download.Send(WGetFile + "-O \"scheduler/scheduler_config.json\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/scheduler/scheduler_config.json");

            Download.Send(WGetFile + "-O \"text_encoder/model.onnx\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/resolve/main/text_encoder/model.onnx");

            Download.Send(WGetFile + "-O \"tokenizer/vocab.json\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/tokenizer/vocab.json");
            Download.Send(WGetFile + "-O \"tokenizer/merges.txt\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/tokenizer/merges.txt");
            Download.Send(WGetFile + "-O \"tokenizer/special_tokens_map.json\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/tokenizer/special_tokens_map.json");
            Download.Send(WGetFile + "-O \"tokenizer/tokenizer_config.json\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/tokenizer/tokenizer_config.json");

            Download.Send(WGetFile + "-O \"unet/weights.pb\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/resolve/main/unet/weights.pb");
            Download.Send(WGetFile + "-O \"unet/model.onnx\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/resolve/main/unet/model.onnx");

            Download.Send(WGetFile + "-O \"vae/model.onnx\" https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/resolve/main/vae/model.onnx");
            Download.Send(WGetFile + "https://huggingface.co/ssube/stable-diffusion-x4-upscaler-onnx/raw/main/model_index.json");
        }
        public static void DownloadCNPoser()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/OpenposeDetector/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }

            Directory.CreateDirectory(WorkingDir);
            Directory.CreateDirectory(WorkingDir + "anannotator");
            Directory.CreateDirectory(WorkingDir + "anannotator\\ckpts");

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();
            Download.Send(WGetFile + "-O \"anannotator\\ckpts\\body_pose_model.pth\" https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth");
            Download.Send(WGetFile + "-O \"anannotator\\ckpts\\network-bsds500.pth\" https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth");
            Download.Send(WGetFile + "-O \"anannotator\\ckpts\\mlsd_large_512_fp32.pth\" https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth");
            Download.SendExitCommand();
            Download.Wait();
        }
        public static void DownloadSDCNPoser()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-openpose/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-openpose/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-openpose-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }

        public static void DownloadSDCNCanny()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-canny/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-canny/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-canny-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }

        public static void DownloadSDCNDepth()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-depth/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-depth/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-depth-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }

        public static void DownloadSDCNHed()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-hed/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-hed/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-hed/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-hed-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }

        public static void DownloadSDCNNormal()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-normal/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-normal/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-normal-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }
        public static void DownloadSDCNScribble()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-scribble/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-scribble/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-scribble-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }
        public static void DownloadSDCNSeg()
        {
            string WorkingDir = FS.GetModelDir() + "controlnet/sd-controlnet-seg/";

            if (Directory.Exists(WorkingDir))
            {
                return;
            }
            Directory.CreateDirectory(WorkingDir);

            string WGetFile = "\"" + FS.GetToolsDir() + "wget.exe\" ";
            Host Download = new Host(WorkingDir);
            Download.Start();

            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-seg/raw/main/config.json");
            Download.Send(WGetFile + "https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/diffusion_pytorch_model.bin");
            Download.Send(WGetFile + "https://huggingface.co/ForserX/sd-controlnet-seg-onnx/resolve/main/model.onnx");
            Download.SendExitCommand();
            Download.Wait();
        }
    }
}
