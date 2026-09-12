using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

static class Program
{
    [STAThread]
    static void Main()
    {
        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
        if (!File.Exists(python))
            python = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");
        string script = Path.Combine(baseDir, "gui_qt.py");

        if (!File.Exists(python) || !File.Exists(script))
        {
            MessageBox.Show(
                "No se encontró el entorno de transcripción en:\n" + baseDir +
                "\n\nAbre una terminal en esa carpeta y ejecuta: uv run gui_qt.py",
                "Transcripción de videos (Qt)", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = python;
        psi.Arguments = "\"" + script + "\"";
        psi.WorkingDirectory = baseDir;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        try
        {
            Process.Start(psi);
        }
        catch (Exception ex)
        {
            MessageBox.Show("Error al iniciar: " + ex.Message,
                "Transcripción de videos (Qt)", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
