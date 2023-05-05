using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Reflection.PortableExecutable;
using System.Runtime.CompilerServices;
using System.Security.Policy;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using Downloader;

namespace SD_FXUI.Utils
{
    public static class FileDownloader
    {
        private static DownloadConfiguration downloadOpt = null;
        private static DownloadService downloader = null;
        private static float progressStatusNextCheck = 0;
        public static void InitDownloaderConfig()
        {
            downloadOpt = new DownloadConfiguration()
            {
                // usually, hosts support max to 8000 bytes, default values is 8000
                BufferBlockSize = 4096,
                // file parts to download, default value is 1
                ChunkCount = 8,
                // download speed limited to 2MB/s, default values is zero or unlimited
                //MaximumBytesPerSecond = 1024 * 1024 * 2,
                // the maximum number of times to fail
                MaxTryAgainOnFailover = 5,
                // download parts of file as parallel or not. Default value is false
                ParallelDownload = true,
                // number of parallel downloads. The default value is the same as the chunk count
                ParallelCount = 8,
                // timeout (millisecond) per stream block reader, default values is 1000
                Timeout = 1000,
                // set true if you want to download just a specific range of bytes of a large file
                RangeDownload = false,
                // floor offset of download range of a large file
                RangeLow = 0,
                // ceiling offset of download range of a large file
                RangeHigh = 0,
                // clear package chunks data when download completed with failure, default value is false
                ClearPackageOnCompletionWithFailure = true,
                // minimum size of chunking to download a file in multiple parts, default value is 512
                MinimumSizeOfChunking = 1024,
                // Before starting the download, reserve the storage space of the file as file size, default value is false
                ReserveStorageSpaceBeforeStartingDownload = true,
                // config and customize request headers
                RequestConfiguration =
                {
                    Accept = "*/*",
                    //CookieContainer = cookies,
                    Headers = new WebHeaderCollection(), // { your custom headers }
                    KeepAlive = true, // default value is false
                    ProtocolVersion = HttpVersion.Version11, // default value is HTTP 1.1
                    UseDefaultCredentials = false,
                    // your custom user agent or your_app_name/app_version.
                    UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",

                }
            };

            downloader = new DownloadService(downloadOpt);

            //  this.slW.ValueChanged += new System.Windows.RoutedPropertyChangedEventHandler<double>(this.slW_ValueChanged);
            //  private void slH_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)

            downloader.DownloadStarted += OnDownloadStarted;
            //downloader.ChunkDownloadProgressChanged += OnChunkDownloadProgressChanged;
            downloader.DownloadProgressChanged += OnDownloadProgressChanged;
            downloader.DownloadFileCompleted += OnDownloadFileCompleted;

        }

        public static async Task DownloadFileAsync(string url, string FullFilePath)
        {
            await downloader.DownloadFileTaskAsync(url, FullFilePath);
        }

        private static void OnDownloadStarted(object sender, DownloadStartedEventArgs e)
        {            
            Host.Print("Download started: " + downloader.Package.FileName + " from Address: " + downloader.Package.Address + " total file size: " + (downloader.Package.TotalFileSize/1024/1024).ToString() + " MB");
        }

        private static void OnChunkDownloadProgressChanged(object sender, Downloader.DownloadProgressChangedEventArgs e)
        {
            //Console.SetCursorPosition(0, Console.CursorTop - 1);
            //Host.ClearCurrentConsoleLine();            
        }

        private static void OnDownloadProgressChanged(object sender, Downloader.DownloadProgressChangedEventArgs e)
        {
            //Console.SetCursorPosition(0, Console.CursorTop - 1);
            //Host.ClearCurrentConsoleLine();
            if(downloader.Package.SaveProgress >= progressStatusNextCheck + 5.0f)
            {
                progressStatusNextCheck += 5.0f;
                Host.Print("DownloadProgress: " + downloader.Package.SaveProgress.ToString().Substring(0,6) + "%");
            }
            
        }

        private static void OnDownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            progressStatusNextCheck = 0;
            Host.Print("OnDownloadFileCompleted");
        }

    }
}

        
   