using SD_FXUI;
using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Controls.Ribbon;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;


namespace SD_FXUI
{
    public partial class App : System.Windows.Application
    {

        private bool _contentLoaded;

        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "7.0.5.0")]
        public void InitializeComponent()
        {
            if (_contentLoaded)
            {
                return;
            }
            _contentLoaded = true;

            this.StartupUri = new System.Uri("MainWindow.xaml", System.UriKind.Relative);

           //System.Uri resourceLocater = new System.Uri("/HandyControl;component/Themes/SkinDefault.xaml", System.UriKind.Relative);
           //System.Uri resourceLocater2 = new System.Uri("/HandyControl;component/Themes/Theme.xaml", System.UriKind.Relative);
           //
           //System.Windows.Application.LoadComponent(this, resourceLocater);
           //System.Windows.Application.LoadComponent(this, resourceLocater2);
        }

        /// <summary>
        /// Application Entry Point.
        /// </summary>
        [System.STAThreadAttribute()]
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "7.0.5.0")]
        public static void Main()
        {
            AppDomain.CurrentDomain.UnhandledException += DumpUtils.CurrentDomain_UnhandledException;
            SD_FXUI.App app = new SD_FXUI.App();
            app.InitializeComponent();
            app.Run();
        }
    }
}

