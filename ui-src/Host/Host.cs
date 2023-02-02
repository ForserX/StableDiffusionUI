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

        public Host(string Dir, string FileName = "cmd.exe")
        {
            ProcessStartInfo = new ProcessStartInfo(FileName);
            ProcessStartInfo.RedirectStandardInput = true;
            ProcessStartInfo.WorkingDirectory = Dir;
            ProcessStartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            ProcessStartInfo.CreateNoWindow = true;
            ProcessStartInfo.UseShellExecute = false;

            ProcessStartInfo.RedirectStandardOutput = true;
            ProcessStartInfo.RedirectStandardError = true;
        }

        public void Start(string arg = "")
        {
            ProcessStartInfo.Arguments = arg;

            Process = Process.Start(ProcessStartInfo);
            Process.OutputDataReceived += new DataReceivedEventHandler((sender, e) =>
            {
                // Prepend line numbers to each line of the output.
                if (!String.IsNullOrEmpty(e.Data))
                {
                    Helper.UIHost.Print(e.Data);
                }
            });

            Process.ErrorDataReceived += new DataReceivedEventHandler((sender, e) =>
            {
                // Prepend line numbers to each line of the output.
                if (!String.IsNullOrEmpty(e.Data))
                {
                    Helper.UIHost.Print(e.Data);
                }
            });

            Process.BeginOutputReadLine();
            Process.BeginErrorReadLine();

            Helper.SecondaryProcessList.Add(this);
        }

        public void Print(string Message)
        {
            Helper.UIHost.Print(Message);
        }
        public void Kill()
        {
            Process.Kill();
        }

        public void Send(string cmd)
        {
            Process.StandardInput.WriteLine("@" + cmd);
        }
        public void SendExistCommand()
        {
            Send("@exit");
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
