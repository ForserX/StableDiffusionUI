using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SD_FXUI
{
    internal class HostFilter
    {
        public static void CheckTraceBack(string Message)
        {
            if (Message.Contains("Traceback (most recent call last)"))
            {
                Helper.UIHost.Dispatcher.Invoke(() => { Helper.UIHost.Show(); });
                Wrapper.SendNotification("Error! See host for details!");
            }
        }

        public static void CheckConvertState(string Message)
        {
            if (Message.Contains("SD: Done"))
            {
                Wrapper.SendNotification(Message);
                Helper.Form.InvokeUpdateModelsList();
            }
        }

        public static void CheckImageState(string Message)
        {
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
        }

        public static void CheckDeepDanBooru(string Message)
        {
            if (Message.Contains("DeepDanBooru: Finished!"))
            {
                string NewPrompt = Message;
                NewPrompt = NewPrompt.Replace("\'", string.Empty);
                NewPrompt = NewPrompt.Replace("[", string.Empty);
                NewPrompt = NewPrompt.Replace("]", string.Empty);
                NewPrompt = NewPrompt.Replace("DeepDanBooru: Finished!", string.Empty);
                NewPrompt = NewPrompt.Replace("rating:safe", string.Empty);

                //NewPrompt = NewPrompt.Substring(1, NewPrompt.Length - 2);
                Helper.Form.InvokeSetPrompt(NewPrompt);
            }
        }
    }
}
