using Microsoft.Toolkit.Uwp.Notifications;
using System.Windows;

namespace SD_FXUI
{
    internal class Notification
    {
        public static void SendNotification(string Message = "Generation completed", string Caption = "Stable Diffusion: FX UI")
        {
            new ToastContentBuilder()
            .AddArgument("action", "viewConversation")
            .AddArgument("conversationId", 9874)
            .AddText(Caption)
            .AddText(Message)
            .Show();
        }

        public static bool MsgBox(string text, string caption = "Info")
        {
            return
                MessageBox.Show(text, caption, MessageBoxButton.OKCancel) == MessageBoxResult.OK;
        }
    }
}
