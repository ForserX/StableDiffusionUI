using System.IO;

namespace SD_FXUI.Utils.Models
{
    internal class TextEncoderONNX
    {
        static public void CheckEncoder()
        {
            if (GlobalVariables.Mode == Helper.ImplementMode.ONNX)
            {
                if (!File.Exists(GetModel()))
                {
                    WGetDownloadModels.DownloadTextEncoder();
                }
            }
        }
        
        static public string GetModel()
        {
            return FS.GetModelDir() + "text_encoder/model.onnx";
        }
        static public string GetModelDir()
        {
            return FS.GetModelDir() + "text_encoder/";
        }

    }
}
