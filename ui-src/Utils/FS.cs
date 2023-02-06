using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class FS
    {
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

        public static string GetWorkingDir()
        {
            return System.IO.Directory.GetCurrentDirectory();
        }
        public static string GetImagesDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\images\\";
        }
        public static string GetModelDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\models\\";
        }
        public static string GetToolsDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\tools\\";
        }
        public static void CopyDirectory(string sourceDir, string destinationDir, bool recursive)
        {
            // Get information about the source directory
            var dir = new DirectoryInfo(sourceDir);

            // Check if the source directory exists
            if (!dir.Exists)
                throw new DirectoryNotFoundException($"Source directory not found: {dir.FullName}");

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
                    CopyDirectory(subDir.FullName, newDestinationDir, true);
                }
            }
        }

        public static List<string> GetModels(Helper.ImplementMode Mode)
        {
            string WorkingPath = GetModelDir();

            switch (Mode)
            {
                case Helper.ImplementMode.ONNX: WorkingPath += "onnx/"; break;
                case Helper.ImplementMode.InvokeAI: WorkingPath += "stable-diffusion/"; break;
                case Helper.ImplementMode.Shark: WorkingPath += "diff/"; break;
                case Helper.ImplementMode.DiffCUDA: WorkingPath += "diff/"; break;
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
                    Models.Add(Path.GetFileName(LocPath));
                }

                if (Mode == Helper.ImplementMode.Shark)
                {
                    WorkingPath = GetModelDir() + @"huggingface/";

                    foreach (string file in Directory.EnumerateFiles(WorkingPath, "*.hgf", SearchOption.AllDirectories))
                    {
                        string UrlName = Path.GetFileName(file).Replace("(slash)", "/");
                        UrlName = UrlName.Remove(UrlName.Length - 4);
                        Models.Add(UrlName);
                    }
                }
            }

            return Models;
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
        }
    }
}