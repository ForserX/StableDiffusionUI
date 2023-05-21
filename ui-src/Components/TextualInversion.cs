using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static SD_FXUI.Helper;

namespace SD_FXUI
{
    class TextualInversion
    {
        static Dictionary<string, string> TIs = null;
        static string[] Formats = new string[] { "safetensors", "ckpt", "pt" };

        public static void Reload()
        {
            if (TIs == null)
            {
                TIs = new Dictionary<string, string>();
            }
            else
            {
                TIs.Clear();
            }

            foreach (string File in FS.GetFilesFrom(FS.GetModelDir(FS.ModelDirs.TextualInversion), new string[] { "safetensors", "ckpt", "pt" }, true))
            {
                string Ext = System.IO.Path.GetExtension(File);
                string FileName = File.Replace(Ext, string.Empty);

                FileName = FileName.Replace("\\", "/");

                TIs.Add(FileName, Ext);
            }
        }

        static string AppendFormat(string FullPath)
        {
            FullPath = FullPath.Replace("\\", "/");

            string ValidName = FullPath;

            string Ext = System.IO.Path.GetExtension(ValidName);

            if (Ext.Length > 1)
            {
                Ext = Ext[1..];

                if (Formats.Contains(Ext))
                {
                    return ValidName;
                }
            }

            if (!TIs.ContainsKey(FullPath))
            {
                ValidName += ".pt";
            }

            ValidName += TIs[FullPath];
            return ValidName;
        }

        public static string Extract(string Prompt, bool IsPositive = true)
        {
            string SourcePrompt = Prompt;
            while (SourcePrompt.Contains("<ti:", 0))
            {
                int StartIdx = SourcePrompt.IndexOf("<", 0);
                int EndIdx = SourcePrompt.IndexOf(">", 0) + 1;

                string DataStr = SourcePrompt.Substring(StartIdx, EndIdx - StartIdx);
                SourcePrompt = SourcePrompt.Replace(DataStr, string.Empty);

                DataStr = DataStr.Replace("<ti:", string.Empty).Replace(">", string.Empty);
                if (!DataStr.Contains(":", 0))
                    continue;

                int DelimerIdx = DataStr.IndexOf(":", 0);

                Helper.LoRAData TIData = new Helper.LoRAData();
                TIData.Name = string.Concat(FS.GetModelDir(FS.ModelDirs.TextualInversion), DataStr.AsSpan(0, DelimerIdx));

                // Validate 
                TIData.Name = AppendFormat(TIData.Name);

                TIData.Value = (float)int.Parse(DataStr[(DelimerIdx + 1)..]) / 100.0f;

                if (IsPositive)
                {
                    Helper.MakeInfo.TI.Add(TIData);
                }
                else
                {
                    Helper.MakeInfo.TINeg.Add(TIData);
                }
            }

            while (SourcePrompt.StartsWith(",") || SourcePrompt.StartsWith(" "))
            {
                SourcePrompt = SourcePrompt.Substring(1);
            }

            return SourcePrompt;
        }
    }
}
