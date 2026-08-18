package BoostNSift;/*
Free to use, change for research purposes.
 */

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.*;
import java.nio.file.FileSystems;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


/**
 *
 * @author Abdul.Razzaq, us
 * The BoostNSift.BoostNSift without CodeSifter
 */
public class BoostN {
    static String indexFilePath = "temp_data/Math/64b/index";
    static String corpusFilePath = "temp_data/Math/64b/Math-64-AfterSplitStopStem.corpusRawMethodLevelGranularity";
    static String titleFilePath = "temp_data/Math/64b/Math-64_Titles-AfterSplitStopStem.csv";
    static String descFilePath = "temp_data/Math/64b/Math-64_Desc-AfterSplitStopStem.csv";
    static String commFilePath = "temp_data/Math/64b/Math-64_Comm-AfterSplitStopStem.csv";
    static String resultFilePath = "temp_data/Math/64b/Results.csv";

//    static HashMap<Integer, Float> grandResults = new HashMap<Integer, Float>();
    public static void main(String[] args) throws IOException, ParseException {
        if (args.length == 2) {
            String docDir = args[0];
            String filePrefix = args[1];
            indexFilePath = docDir + "/index";
            corpusFilePath = docDir + "/" + filePrefix + "-AfterSplitStopStem.corpusRawMethodLevelGranularity";
            titleFilePath = docDir + "/" + filePrefix + "_Titles-AfterSplitStopStem.csv";
            descFilePath = docDir + "/" + filePrefix + "_Desc-AfterSplitStopStem.csv";
            commFilePath = docDir + "/" + filePrefix + "_Comm-AfterSplitStopStem.csv";
            resultFilePath = docDir + "/" + "Results.csv";
        }
        //Following statement might throw error for the first time when index directory does not exist
        File indexFile = new File(indexFilePath);
        if (!indexFile.exists())
            indexFile.mkdirs();
        FileUtils.cleanDirectory(new File(indexFilePath));
        Analyzer analyzer = new StandardAnalyzer();
        String result = "";
        //--------------------------------------------------------------------------------------------//
        //1: Create a local directory index
        Directory indexDir = FSDirectory.open(FileSystems.getDefault().getPath(indexFilePath));
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        // count from 0 instead of 1!
        int counter = 0;
        IndexWriter iwriter = new IndexWriter(indexDir, config);

        //Method corpus file to read. Each line in file should contain the corpus of a method
        File file = new File(corpusFilePath);

        try (FileReader fileReader = new FileReader(file)) {
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String text;

            while ((text = bufferedReader.readLine()) != null) {
                Document doc = new Document();

                // 这里id完全可以使用StringField
                doc.add(new Field("method_id", Integer.toString(counter), TextField.TYPE_STORED));
                doc.add(new Field("method_contents", text, TextField.TYPE_STORED));

                iwriter.addDocument(doc);
                counter++;
            }
        }
        iwriter.close();
        System.out.println("Counter: "+ counter);
//--------------------------------------------------------------------------------------------//

        //2. Initialize the IR algorithm and Query Parser
        DirectoryReader ireader = DirectoryReader.open(indexDir);
        IndexSearcher isearcher = new IndexSearcher(ireader);
        float k1=1.0f;
        float b=0.3f;

        if(counter>3000)
            isearcher.setSimilarity(new BM25Similarity(k1, b));
        else
            isearcher.setSimilarity(new BM25Similarity(0.0f, b));

        QueryParser parser = new QueryParser("method_contents", analyzer);

        // Read title, descriptions and comments in temporary objects
        File Qfile;
        List<String> titles = new ArrayList<String>();
        Qfile = new File(titleFilePath);
        try (FileReader fileReader = new FileReader(Qfile)) {
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String qry;
            while ((qry = bufferedReader.readLine()) != null) {
                titles.add(qry.trim());
            }
        }
        List<String> descs = new ArrayList<String>();
        Qfile = new File(descFilePath);
        System.out.println(descFilePath);
        String tmp_desc = "";
        try (FileReader fileReader = new FileReader(Qfile)) {
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String qry;

            while ((qry = bufferedReader.readLine()) != null) {
                tmp_desc += qry.trim();
            }
        }
        descs.add(tmp_desc);
        System.out.println(tmp_desc);
        List<String> comments = new ArrayList<String>();
        Qfile = new File(commFilePath);
        try (FileReader fileReader = new FileReader(Qfile)) {
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String qry;

            while ((qry = bufferedReader.readLine()) != null) {
                comments.add(qry.trim());
            }
        }


        File newFile = new File(resultFilePath);
        BufferedWriter writer;
        writer = new BufferedWriter(new FileWriter(newFile));

//--------------------------------------------------------------------------------------------//
        //3. Boost queries after reading: assign weight to each construct of query
        int j=0;
        ArrayList<String> queries= new ArrayList<String>();
//        String AggregatedQuery="";
        for (String desc : descs) {
            // String title = "";
            // if (titles.size() > j)
            //     title=titles.get(j);
            String title=titles.get(j);
            String comment = "";
            if (comments.size() > j)
                comment=comments.get(j);
            j++;
            BooleanQuery.setMaxClauseCount(900000000);

            Query qry;
            if(!title.trim().equals("")){
                qry = parser.parse(title);
                title= qry.toString().trim().replaceAll(" ", "^1 ");
            }
            if(!desc.trim().equals("")){
                qry = parser.parse(desc);
                desc= qry.toString().trim().replaceAll(" ", "^1 ");
            }
            // Since comments are attached in description, here adding comments again mean you are weighting them 2 times higher than descriptions
            if(!comment.trim().equals("")){
                qry = parser.parse(comment);
                comment= qry.toString().trim().replaceAll(" ", "^1 ");
            }

            desc= title.trim()+" "+desc.trim()+" "+comment.trim();
            queries.add(desc);
//             将所有的desc都添加到一起？
//            AggregatedQuery+=desc+" ";
        }

//--------------------------------------------------------------------------------------------//
        //4. Get the results against the aggregated query
//        Query quToSearch = new QueryParser("method_contents", analyzer).parse(AggregatedQuery);
//
//        ScoreDoc[] nhits = isearcher.search(quToSearch,counter, Sort.RELEVANCE, true, false).scoreDocs;
//
//        // Iterate through the grand results:
//        for (int i = 0; i < nhits.length; i++) {
//            Document hitDoc = isearcher.doc(nhits[i].doc);
//            grandResults.put(Integer.parseInt(hitDoc.get("method_id")),nhits[i].score/nhits[0].score);
//        }

//--------------------------------------------------------------------------------------------//
        //5. Now perform sifting of the retrieved results against each query

        j=0;
        for(String quString: queries){
//            System.out.println("running:"+j++);
            result="";
            Query queryToSearch = new QueryParser("method_contents", analyzer).parse(quString);
            ScoreDoc[] hits = isearcher.search(queryToSearch,counter, Sort.RELEVANCE, true, false).scoreDocs;

            // Iterate through the results:

            for (int i = 0; i < hits.length; i++) {
                Document hitDoc = isearcher.doc(hits[i].doc);

                //System.out.println(hits[i].doc);

                if(hitDoc.get("method_contents").length()<11)continue;
//                if(grandResults.get(Integer.parseInt(hitDoc.get("method_id")))!=null){
//                    double d=grandResults.get(Integer.parseInt(hitDoc.get("method_id")));
//                    if(d>(hits[i].score/hits[0].score))
//                        continue;
//                }
//                else
//                    continue;

                result +=  hitDoc.get("method_id") + "," + (hits[i].score/hits[0].score)+",";

            }
            // Store intermediate results to calculate evaluation measures after
            if(result!="")result=result.substring(0, result.length()-1) ;
            writer.write(result + "\n");
            writer.flush();
        }
        writer.close();
        ireader.close();
        indexDir.close();

    }

}
