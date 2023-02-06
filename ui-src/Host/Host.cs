using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Documents;

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
            if (Message.Contains("SD: Done"))
            {
                Wrapper.SendNotification(Message);
            }

            if (Message.Contains("Traceback (most recent call last)"))
            {
                Helper.UIHost.Dispatcher.Invoke(()=> { Helper.UIHost.Show(); });
                Wrapper.SendNotification("Error! See host for details!");
            }

            if (Message.Contains("SD: Model loaded"))
            {
                Helper.Form.InvokeProgressUpdate(20);
            }
            
            if (Message.Contains("SD: Generating done"))
            {
                Helper.Form.InvokeProgressUpdate(60);
            }
            
            if (Message.Contains("Image generate"))
            {
                Helper.Form.InvokeProgressApply();
            }

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
