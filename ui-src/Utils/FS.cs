using System.Collections.Generic;
using System.IO;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using static SD_FXUI.FS;

namespace SD_FXUI
{
    internal class FS
    {
        public enum ModelDirs
        {
            Diffusers,
            ONNX,
            LoRA,
            OpenPose,
            TextualInversion,

            General = -1
        }

        public static string[] GetFilesFrom(string searchFolder, string[] filters, bool isRecursive)
        {
            List<string> filesFound = new List<string>();
            var searchOption = isRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var filter in filters)
            {
                filesFound.AddRange(Directory.GetFiles(searchFolder, string.Format("*.{0}", filter), searchOption));
            }
            return filesFound.ToArray();
        }

        public static string GetModelRootDir()
        {
            bool IsNull = GlobalVariables.ModelsDir == null || GlobalVariables.ModelsDir.Length < 3;
            bool IsNotFound = !Directory.Exists(GlobalVariables.ModelsDir);
            if (IsNull || IsNotFound)
            {
                if (!IsNull && IsNotFound)
                {
                    Notification.MsgBox($"Models directory {GlobalVariables.ModelsDir} not found! Used default path.");
                }

                GlobalVariables.ModelsDir = Directory.GetCurrentDirectory() + "\\models\\";
            }

            return GlobalVariables.ModelsDir;
        }

        public static string GetWorkingDir()
        {
            return Directory.GetCurrentDirectory();
        }
        public static string GetImagesDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\images\\";
        }
        public static string GetModelDir(ModelDirs SubDir = ModelDirs.General)
        {
            string GeneralPath = GetModelRootDir();

            switch (SubDir)
            {
                case ModelDirs.ONNX:
                    GeneralPath += "onnx/";
                    break;

                case ModelDirs.Diffusers:
                    GeneralPath += "diffusers/";
                    break;

                case ModelDirs.LoRA:
                    GeneralPath += "lora/";
                    break;

                case ModelDirs.TextualInversion:
                    GeneralPath += "textual_inversion/";
                    break;

                case ModelDirs.OpenPose:
                    GeneralPath += "OpenPose/";
                    break;
            }

            return GeneralPath;
        }
        public static string GetToolsDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "/tools/";
        }

        public static string GetModelLogoPath(string Model)
        {
            string FirstPath = FS.GetModelDir(ModelDirs.ONNX);

            if (GlobalVariables.Mode != Helper.ImplementMode.ONNX)
                FirstPath = FS.GetModelDir(ModelDirs.Diffusers);

            string TryModelPath = FirstPath + Model;

            if (File.Exists(TryModelPath + "/logo.png"))
            {
                return TryModelPath + "/logo.png";
            }

            if (File.Exists(TryModelPath + "/logo.jpg"))
            {
                return TryModelPath + "/logo.jpg";
            }

            return string.Empty;
        }

        public static List<string> GetModels(Helper.ImplementMode Mode)
        {
            string WorkingPath = GetModelDir();

            switch (Mode)
            {
                case Helper.ImplementMode.ONNX: WorkingPath += "onnx/"; break;
                case Helper.ImplementMode.InvokeAI: WorkingPath += "stable-diffusion/"; break;
                case Helper.ImplementMode.DiffCUDA: WorkingPath += "diffusers/"; break;
                case Helper.ImplementMode.DiffCPU: WorkingPath += "diffusers/"; break;
            }

            List<string> Models = new List<string>();

            if (Mode == Helper.ImplementMode.InvokeAI)
            {
                foreach (string file in Directory.EnumerateFiles(WorkingPath + @"stable-diffusion/", "*.ckpt", SearchOption.AllDirectories))
                {
                    Models.Add(Path.GetFileName(file));
                }
            }
            else
            {
                foreach (var LocPath in Directory.GetDirectories(WorkingPath))
                {
                    if (!File.Exists(LocPath + "/model_index.json"))
                        continue;

                        Models.Add(Path.GetFileName(LocPath));
                }

                if (Mode != Helper.ImplementMode.ONNX)
                {
                    WorkingPath = GetModelDir() + @"huggingface/";

                    foreach (string file in Directory.EnumerateFiles(WorkingPath, "*.hgf", SearchOption.AllDirectories))
                    {
                        string UrlName = Path.GetFileName(file).Replace("(slash)", "/");
                        Models.Add(UrlName);
                    }
                }
            }

            return Models;
        }

        public static bool HasExt(string File, string[] Formats)
        {
            foreach (string Format in Formats)
            {
                if (File.EndsWith(Format))
                {
                    return true;
                }
            }

            return false;
        }

        public static bool IsDirectory(string Path)
        {
            FileAttributes attr = File.GetAttributes(Path);
            return (attr.HasFlag(FileAttributes.Directory));
        }

        internal class Dir
        {
            static public void Delete(string Name, bool Recursive)
            {
                if (Directory.Exists(Name))
                {
                    if (Recursive)
                    {
                        Directory.Delete(Name, true);
                    }
                    else
                    {
                        Directory.Delete(Name);
                    }
                }
            }

            public static void Copy(string sourceDir, string destinationDir, bool recursive)
            {
                // Get information about the source directory
                var dir = new DirectoryInfo(sourceDir);

                // Check if the source directory exists
                if (!dir.Exists)
                {
                    Host.Print($"Source directory not found: {dir.FullName}");
                    return;
                }

                // Cache directories before we start copying
                DirectoryInfo[] dirs = dir.GetDirectories();

                // Create the destination directory
                Directory.CreateDirectory(destinationDir);

                // Get the files in the source directory and copy to the destination directory
                foreach (FileInfo file in dir.GetFiles())
                {
                    string targetFilePath = Path.Combine(destinationDir, file.Name);
                    file.CopyTo(targetFilePath);
                }

                // If recursive and copying subdirectories, recursively call this method
                if (recursive)
                {
                    foreach (DirectoryInfo subDir in dirs)
                    {
                        string newDestinationDir = Path.Combine(destinationDir, subDir.Name);
                        Copy(subDir.FullName, newDestinationDir, true);
                    }
                }
            }

        }
    }
}