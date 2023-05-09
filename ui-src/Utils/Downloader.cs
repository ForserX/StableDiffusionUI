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

        static int PreviousProgress = 0;
        private static void OnDownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            //Console.SetCursorPosition(0, Console.CursorTop - 1);
            //Host.ClearCurrentConsoleLine();
            if(e.ProgressPercentage > PreviousProgress && e.ProgressPercentage % 5 == 0)
            {
                PreviousProgress = e.ProgressPercentage;
                Host.Print("DownloadProgress: " + e.ProgressPercentage.ToString() + $"% ({e.BytesReceived / 1024 / 1024}/{e.TotalBytesToReceive / 1024 / 1024} mb)" );
            }
            
        }

        private static void OnDownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            PreviousProgress = 0;
            TaskCompleted = true;
            Host.Print("OnDownloadFileCompleted");
        }

    }
}

        
   