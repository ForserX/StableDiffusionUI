using System.Collections.Generic;
using System.Windows.Media;

namespace SD_FXUI
{
    internal class Helper
    {
        public enum UpscalerType
        {
            ESRGAN,
            ESRGAN_X4,
            ESRGAN_ANIME,
            ESRGAN_NET,
            WAIFU_CU,
            WAIFU_UP_PHOTO,
            WAIFU_UP_ART,
            SR,
            SRMD,
            None
        }
        public enum DrawingMode
        {
            Text2Img,
            Img2Img,
            Inpaint
        }

        public enum ImplementMode
        {
            DiffCUDA,
            ONNX,
            DiffCPU,
            InvokeAI,

            IDK
        }

        public enum ImageState
        {
            Free,
            Favor
        }

        public enum VENV
        {
            DiffCUDA,
            DiffONNX,
            DiffCPU,
            Any
        }

        public struct LoRAData
        {
            public string Name;
            public float Value;
        }

        public struct MetaInfo
        {
            public string Prompt;
            public string NegPrompt;
            public long StartSeed;
            public int Steps;
            public float CFG;
            public float ImgScale;
            public float ImgCFGScale;
            public string Model;
            public string VAE;
            public string Sampler;
            public int ETA;
            public string Mode;
            public string Image;
            public string Mask;
            public int TotalCount;
            public int Width;
            public int Height;
            public string WorkingDir;
            public string Device;
            public bool fp16;
            public int BatchSize;

            public List<LoRAData> LoRA;
            public List<LoRAData> TI;
            public List<LoRAData> TINeg;
        }

        public static ImageSource getImageSourceUndefined() {return GlobalVariables.NoImageData; }
    }
}
