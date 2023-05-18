using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Media.AppBroadcasting;

namespace SD_FXUI
{
    class LoRA
    {
        static Dictionary<string, string> LoRAs = null;

        public static void Reload()
        {
            if (LoRAs == null)
            {
                LoRAs = new Dictionary<string, string>();
            }
            else
            {
                LoRAs.Clear();
            }

            foreach(string File in FS.GetFilesFrom(FS.GetModelDir(FS.ModelDirs.LoRA), new string[]{ "safetensors", "ckpt", "pt"}, true))
            {
                string Ext = System.IO.Path.GetExtension(File);
                string FileName = File.Replace(Ext, string.Empty);

                FileName = FileName.Replace("\\", "/");

                LoRAs.Add(FileName, Ext);
            }
        }

        static string GetFormat(string FullPath)
        {
            FullPath = FullPath.Replace("\\", "/");

            string ValidName = FullPath;

            if (!LoRAs.ContainsKey(FullPath)) 
            {
                ValidName += ".safetensors";
            }

            ValidName += LoRAs[FullPath];
            return ValidName;
        }

        public static string Extract(string Prompt)
        {
            string SourcePrompt = Prompt;

            while (SourcePrompt.Contains("<lora:", 0))
            {
                int StartIdx = SourcePrompt.IndexOf("<", 0);
                int EndIdx = SourcePrompt.IndexOf(">", 0) + 1;

                string LoraDataStr = SourcePrompt.Substring(StartIdx, EndIdx - StartIdx);
                SourcePrompt = SourcePrompt.Replace(LoraDataStr, string.Empty);

                LoraDataStr = LoraDataStr.Replace("<lora:", string.Empty).Replace(">", string.Empty);
                if (!LoraDataStr.Contains(":", 0))
                    continue;

                int DelimerIdx = LoraDataStr.IndexOf(":", 0);

                Helper.LoRAData LoRAData = new Helper.LoRAData();
                LoRAData.Name = string.Concat(FS.GetModelDir(FS.ModelDirs.LoRA), LoraDataStr.AsSpan(0, DelimerIdx));

                // Validate 
                LoRAData.Name = GetFormat(LoRAData.Name);

                LoRAData.Value = (float)int.Parse(LoraDataStr[(DelimerIdx + 1)..]) / 100.0f;

                Helper.MakeInfo.LoRA.Add(LoRAData);
            }

            return SourcePrompt;
        }

    }
}
