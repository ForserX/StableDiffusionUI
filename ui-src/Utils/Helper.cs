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

        public static List<string> ModelsList = new List<string>();
    }
}
