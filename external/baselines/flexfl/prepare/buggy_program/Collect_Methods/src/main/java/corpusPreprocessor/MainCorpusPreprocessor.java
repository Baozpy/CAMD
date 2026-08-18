package corpusPreprocessor;


public class MainCorpusPreprocessor {
  static void testjEdit() throws Exception {
    String inputFileNameCorpus = "TestCases/Input/Corpus-jEdit4.3.corpusRawMethodLevelGranularity";
    String outputFolder = "TestCases/Output/";
    CorpusPreprocessor corpusPreprocessor = new CorpusPreprocessor(inputFileNameCorpus, outputFolder);
    corpusPreprocessor.preprocessCorpus();
  }
  
  static void testSystem1(String inputFileName, String outputFolder) throws Exception {
    CorpusPreprocessor corpusPreprocessor = new CorpusPreprocessor(inputFileName, outputFolder);
    corpusPreprocessor.preprocessCorpus();
  }
  
  static void testSystem2() throws Exception {
    String inputFileNameCorpus = "TestCases/Input/Corpus-System2.corpusRawMethodLevelGranularity";
    String outputFolder = "TestCases/Output/";
    CorpusPreprocessor corpusPreprocessor = new CorpusPreprocessor(inputFileNameCorpus, outputFolder);
    corpusPreprocessor.preprocessCorpus();
  }
  
  static void testSystem3() throws Exception {
    String inputFileNameCorpus = "TestCases/Input/Corpus-System3.corpusRawMethodLevelGranularity";
    String outputFolder = "TestCases/Output/";
    CorpusPreprocessor corpusPreprocessor = new CorpusPreprocessor(inputFileNameCorpus, outputFolder);
    corpusPreprocessor.preprocessCorpus();
  }
  
  public static void main(String[] args) throws Exception {
    // outputFolder/file_prefix.corpusRawMethodLevelGranularity
    // outputFolder/file_prefix_Desc.csv
    // outputFolder/file_prefix_Titles.csv
    // outputFolder/file_prefix_Comm.csv
    // String inputFileNameCorpus = "temp_data/Math/64b/Math-64.corpusRawMethodLevelGranularity";
    // "flattened_bug_report/Math-64_Comm.csv"
    // "temp_data/Math/64b"
    testSystem1(args[0], args[1]);
  }
}
