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
        public static String[] GetFilesFrom(String searchFolder, String[] filters, bool isRecursive)
        {
            List<String> filesFound = new List<String>();
            var searchOption = isRecursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var filter in filters)
            {
                filesFound.AddRange(Directory.GetFiles(searchFolder, String.Format("*.{0}", filter), searchOption));
            }
            return filesFound.ToArray();
        }

        public static string GetWorkingDir()
        {
            return System.IO.Directory.GetCurrentDirectory();
        }
        public static string GetModelDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\models\\";
        }
        public static string GetToolsDir()
        {
            return System.IO.Directory.GetCurrentDirectory() + "\\tools\\";
        }
    }
}
