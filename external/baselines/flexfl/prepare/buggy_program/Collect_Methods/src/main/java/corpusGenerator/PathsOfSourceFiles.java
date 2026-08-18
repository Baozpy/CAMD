package corpusGenerator;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import org.apache.commons.io.FileUtils;

public class PathsOfSourceFiles {

	  public static void main(String[] args) throws IOException {
		  String sourceListFile = args[0];
		  String sourceFolderPath = args[1];
//		  File sourceFile = new File("temp_data/Math/64b/SourcePaths.txt");
		  File sourceFile = new File(sourceListFile);
		  File srcDir = new File(sourceFile.getParent());
		  if (!srcDir.exists())
			  srcDir.mkdirs();

	    FileWriter writer = new FileWriter(sourceFile);

		File dir = new File(sourceFolderPath);
	    String[] extensions = { "java" };
	    List<File> files = (List<File>)FileUtils.listFiles(dir, extensions, true);
	    for (File file : files)
	      writer.append(file.getCanonicalPath() + "\r\n"); 
	    writer.flush();
	  }
	

}
