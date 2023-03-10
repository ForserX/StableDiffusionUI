using System.Collections.Generic;
using System.IO;
using System.Windows.Media.Imaging;
using System.Windows.Media;

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
                    if (Mode != Helper.ImplementMode.ONNX)
                        Models.Add(Path.GetFileName(LocPath));
                    else
                    {
                        if (!LocPath.EndsWith("_cn"))
                        {
                            // Skip control net prepared models
                            Models.Add(Path.GetFileName(LocPath));
                        }
                    }
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

        public static ImageSource BitmapFromUri(System.Uri source)
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = source;
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.EndInit();
            return bitmap;
        }

        public static string MetaData(string File)
        {
            string MetaText = "No meta";
            var Data = MetadataExtractor.ImageMetadataReader.ReadMetadata(File);
            if (Data[1].Tags[0].Description != null)
            {
                MetaText = Data[1].Tags[0].Description;
                MetaText = MetaText.Replace("XUI Metadata: ", string.Empty);
            }

            return MetaText;
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