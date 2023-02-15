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
                    if (!String.IsNullOrEmpty(e.Data))
                    {
                        Print(e.Data);
                    }
                });

                Process.ErrorDataReceived += new DataReceivedEventHandler((sender, e) =>
                {
                    // Prepend line numbers to each line of the output.
                    if (!String.IsNullOrEmpty(e.Data))
                    {
                        Print(e.Data);
                    //    Helper.UIHost.Dispatcher.Invoke(()=> Helper.UIHost.Show());
                    }
                });

                Process.BeginOutputReadLine();
                Process.BeginErrorReadLine();
            }

            Helper.SecondaryProcessList.Add(this);
        }

        static public void Print(string Message)
        {
            HostFilter.CheckTraceBack(Message);
            HostFilter.CheckConvertState(Message);
            HostFilter.CheckImageState(Message);
            HostFilter.CheckDeepDanBooru(Message);

            Helper.UIHost.Print(Message);
        }
        public void Kill()
        {
            Process.Kill();
        }

        public void Send(string cmd)
        {
            Process.StandardInput.WriteLine(cmd);
        }
        /*
        public string ReadFrom()
        {
            return Process.StandardOutput.ReadToEnd();
        }
        */

        public void SendExitCommand()
        {
            Send("exit");
        }

        public void Wait()
        {
            Process.WaitForExit();
            if (Helper.SecondaryProcessList.Contains(this))
            {
                Helper.SecondaryProcessList.Remove(this);
            }
        }
    }
}
