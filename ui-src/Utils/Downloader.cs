using System;
using System.ComponentModel;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace SD_FXUI.Utils
{
    public static class FileDownloader
    {
        private static WebClient Client = new WebClient();
        private static bool TaskCompleted = false;
        private static float progressStatusNextCheck = 0;

        public static void Initial()
        {
            Client.DownloadFileCompleted += OnDownloadFileCompleted;
            Client.DownloadProgressChanged += OnDownloadProgressChanged;
        }

        public static async Task DownloadFileAsync(string url, string FullFilePath, bool Wait = true)
        {
            if (!System.IO.File.Exists(url))
            {
                TaskCompleted = false;
                Host.Print("Download started: " + FullFilePath);
                Client.DownloadFileAsync(new Uri(url), FullFilePath);

                if (Wait)
                {
                    FileDownloader.Wait();
                }
            }
        }

        public static void Wait()
        {
            while (!TaskCompleted)
                Thread.Sleep(10);
        }

        private static void OnDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            //Console.SetCursorPosition(0, Console.CursorTop - 1);
            //Host.ClearCurrentConsoleLine();
            if(e.ProgressPercentage >= progressStatusNextCheck + 5.0f)
            {
                progressStatusNextCheck += 5.0f;
                Host.Print("DownloadProgress: " + e.ProgressPercentage.ToString() + $"% ({e.BytesReceived / 1024 / 1024}/{e.TotalBytesToReceive / 1024 / 1024} mb)" );
            }
            
        }

        private static void OnDownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            progressStatusNextCheck = 0;
            TaskCompleted = true;
            Host.Print("OnDownloadFileCompleted");
        }

    }
}

        
   