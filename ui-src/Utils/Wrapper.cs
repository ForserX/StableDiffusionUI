using System;
using System.Runtime.InteropServices;

namespace SD_FXUI
{
    class Wrapper
    {
        public const int GWL_STYLE = -16;
        public const int WS_SYSMENU = 0x80000;
        public const int WS_CHILDWINDOW	= 0x40000;
        [DllImport("user32.dll", SetLastError = true)]
        public static extern int GetWindowLong(IntPtr hWnd, int nIndex);
        [DllImport("user32.dll")]
        public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);


        [DllImport("dbghelp.dll")]
        public static extern bool MiniDumpWriteDump(IntPtr hProcess, int processId, IntPtr hFile, int dumpType,
            IntPtr exceptionParam, IntPtr userStreamParam, IntPtr callStackParam);

    }
}
