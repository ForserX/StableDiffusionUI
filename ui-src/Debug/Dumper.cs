using SD_FXUI.Debug;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SD_FXUI
{
    /// <summary>
    /// Minidump support tools.
    /// </summary>
   [Flags]
    public enum MinidumpType
    {
        MiniDumpNormal = 0x00000000,
        MiniDumpWithDataSegs = 0x00000001,
        MiniDumpWithFullMemory = 0x00000002,
        MiniDumpWithHandleData = 0x00000004,
        MiniDumpFilterMemory = 0x00000008,
        MiniDumpScanMemory = 0x00000010,
        MiniDumpWithUnloadedModules = 0x00000020,
        MiniDumpWithIndirectlyReferencedMemory = 0x00000040,
        MiniDumpFilterModulePaths = 0x00000080,
        MiniDumpWithProcessThreadData = 0x00000100,
        MiniDumpWithPrivateReadWriteMemory = 0x00000200,
        MiniDumpWithoutOptionalData = 0x00000400,
        MiniDumpWithFullMemoryInfo = 0x00000800,
        MiniDumpWithThreadInfo = 0x00001000,
        MiniDumpWithCodeSegs = 0x00002000,
        MiniDumpWithoutAuxiliaryState = 0x00004000,
        MiniDumpWithFullAuxiliaryState = 0x00008000,
        MiniDumpWithPrivateWriteCopyMemory = 0x00010000,
        MiniDumpIgnoreInaccessibleMemory = 0x00020000,
        MiniDumpWithTokenInformation = 0x00040000
    };

    public static class DumpUtils
    {
        /// <summary>
        /// Folder for saved minidumps.
        /// </summary>
        public const string DumpDirectory = "Minidump";

        /// <summary>
        /// Write minidump to file.
        /// </summary>
        /// <param name="minidumpType">Minidump flag(s).</param>
        [MethodImpl(MethodImplOptions.NoOptimization)]
        public static bool WriteDump(
            MinidumpType minidumpType = MinidumpType.MiniDumpWithFullMemory |
            MinidumpType.MiniDumpWithHandleData |
            MinidumpType.MiniDumpWithUnloadedModules |
            MinidumpType.MiniDumpWithFullMemoryInfo |
            MinidumpType.MiniDumpWithThreadInfo)
        {
            try
            {
                if (!Directory.Exists(DumpDirectory))
                {
                    Directory.CreateDirectory(DumpDirectory);
                }

                var currentProcess = Process.GetCurrentProcess();
                var fileName = GetNewDumpFileName(currentProcess.ProcessName);
                var currentDir = FS.GetWorkingDir();
                var filePath = Path.Combine(currentDir, DumpDirectory, fileName);
                var handler = currentProcess.Handle;
                var processId = currentProcess.Id;

                using (var fileStream = new FileStream(filePath, FileMode.CreateNew))
                {
                    return Wrapper.MiniDumpWriteDump(
                        handler,
                        processId,
                        fileStream.SafeFileHandle.DangerousGetHandle(),
                        (int)minidumpType,
                        IntPtr.Zero,
                        IntPtr.Zero,
                        IntPtr.Zero);
                }
            }
            catch (Exception)
            {
                return false;
            }
        }
        public static void CurrentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            string CallStack = e.ExceptionObject.ToString();

            try
            {
                Log.SendMessageToFile(CallStack);
            }
            catch
            {
                // So sad...
            }

            Notification.MsgBox(CallStack);
            WriteDump();
        }

        private static string GetNewDumpFileName(string processName)
        {
            return string.Format("{0}_{1}_{2}.dmp", processName,
                DateTime.Now.ToString("yyyy-dd-mm_HH-mm-ss"),
                Path.GetRandomFileName().Replace(".", ""));
        }
    }
}