using SD_FXUI.Utils;
using System;
using System.Diagnostics;

namespace SD_FXUI
{
    class Host
    {
        Process? Process = null;
        ProcessStartInfo? ProcessStartInfo = null;
        bool NeedDraw = false;

        public Host(string Dir, string FileName = "cmd.exe", bool Show = false)
        {
            ProcessStartInfo = new ProcessStartInfo(FileName);
            ProcessStartInfo.RedirectStandardInput = true;
            ProcessStartInfo.WorkingDirectory = Dir;

            if(!Show)
            {
                ProcessStartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                ProcessStartInfo.CreateNoWindow = true;
                ProcessStartInfo.UseShellExecute = false;

                ProcessStartInfo.RedirectStandardOutput = true;
                ProcessStartInfo.RedirectStandardError = true;
            }
            NeedDraw = Show;
        }

        public void Start(string arg = "")
        {
            ProcessStartInfo.Arguments = arg;

            Process = Process.Start(ProcessStartInfo);

            if (!NeedDraw)
            {
                Process.OutputDataReceived += new DataReceivedEventHandler((sender, e) =>
                {
                    // Prepend line numbers to each line of the output.
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Print(e.Data);
                    }
                });

                Process.ErrorDataReceived += new DataReceivedEventHandler((sender, e) =>
                {
                    // Prepend line numbers to each line of the output.
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        Print(e.Data);
                    //    Helper.UIHost.Dispatcher.Invoke(()=> Helper.UIHost.Show());
                    }
                });

                Process.BeginOutputReadLine();
                Process.BeginErrorReadLine();
            }

            GlobalVariables.SecondaryProcessList.Add(this);
        }

        static public void Print(string Message)
        {
            HostFilter.CheckTraceBack(Message);
            HostFilter.CheckConvertState(Message);
            HostFilter.CheckImageState(Message);
            HostFilter.CheckControlNet(Message);
            HostFilter.CheckDeepDanBooru(Message);
            HostFilter.CheckImageSize(Message);
            HostFilter.CheckOutOfMemory(Message);
            HostFilter.CheckHalfPrecision(Message);

            if (HostFilter.CheckFalseWarning(Message))
                return;

            GlobalVariables.UIHost.Print(Message);

        }

        public static void ClearCurrentConsoleLine()
        {
            int currentLineCursor = Console.CursorTop;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new string(' ', Console.WindowWidth)); 
            Console.SetCursorPosition(0, currentLineCursor);
        }


        public void Kill()
        {
            Process.Kill();
        }

        public void Send(string cmd)
        {
            Process.StandardInput.WriteLine(cmd);
        }

        public void SendExitCommand()
        {
            Send("exit");
        }

        public void Wait()
        {
            Process.WaitForExit();
            if (GlobalVariables.SecondaryProcessList.Contains(this))
            {
                GlobalVariables.SecondaryProcessList.Remove(this);
            }
        }
    }
}
