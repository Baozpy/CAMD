package groundTruthBuilder;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import configuration.ConfigurationParameters;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class GroundTruthBuilder {
    public static void main(String[] args) throws IOException {
        Gson gson = new Gson();


        ConfigurationParameters.setParametersFromSettingsFile();
        List<String> allDefects = Files.readAllLines(Paths.get(ConfigurationParameters.bugListFile));
        Set<String> defectSet = new HashSet<>(allDefects);

        Type bugList = new TypeToken<ArrayList<NL2FixBug>>(){}.getType();
        List<NL2FixBug> bugs = gson.fromJson(new FileReader("data/nl2fix_bugs.json"), bugList);

        for (NL2FixBug bug : bugs) {
            if (!defectSet.contains(bug.bug_id))
                continue;
            // 首先利用BoostNSift的脚本从相应的文件中提取method
            // 获得signature跟target signature相等的method
            // 如果只有一个，那么直接返回
            // 如果有多个，那么找出有交集的；
            // assert 有交集的只有一个
            String methodSig = MethodFinder.extractMethodSignature(bug.method);
            String repoPath = ConfigurationParameters.getDefectRepoPath(bug.bug_id);
            String filePath = repoPath + "/" + bug.source_file;
            String fileContent = new String(Files.readAllBytes(Paths.get(filePath)), StandardCharsets.UTF_8);
            MethodFinder finder = new MethodFinder(fileContent);
            bug.signature = finder.findMethod(methodSig);
        }

        try (FileWriter writer = new FileWriter("data/nl2fix_bug_signatures.json")){
            gson.toJson(bugs, writer);
        }
    }
}
