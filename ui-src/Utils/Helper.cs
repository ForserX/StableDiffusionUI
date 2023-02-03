using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class Helper
    {
        public enum UpscalerType
        {
            ESRGAN,
            ESRGAN_ANIME,
            SR,
            SRMD,
            None
        }

        public enum ImplementMode
        {
            InvokeAI,
            Shark,
            ONNX
        }

        public static List<Host> SecondaryProcessList = new List<Host>();

        public static string CachePath = string.Empty;
        public static string ImgPath = string.Empty;
        public static MainWindow Form = null;
        public static HostForm UIHost = null;
        public static ImplementMode Mode = ImplementMode.Shark;
    }
}
