/*
# MIT License
#
# Copyright (c) 2022 LASER-UMASS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
*/

package configuration;

import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class ConfigurationParameters {
	protected static Logger logger = Logger.getLogger(ConfigurationParameters.class);
		
	
	public static final int k = 50;				// number of suspicious files to be considered
	public static final int All = 10000;		// to fetch all suspicious statements of a suspicious file
	public static final int[] m = new int[]{1, 25, 50, 100, All};		// number of suspicious statements per suspicious file	
	public static final String[] scoringStrategy = new String[] {"high", "wted"};	// score function
	public static String rootDirectory = null;		// absolute path to the cloned Blues directory (specify this in the blues.settings file)
	public static String defects4JHome = null;		// path to defects4j 
	public static String IndriPath = null;			// path to Indri toolkit
	public static String stopwordsFile = null; 				// file listing stopwords
	public static String fieldsFile = null;					// file listing fields extracted from source file documents
	public static String bugCheckoutDirectory = null;		// directory to checkout defects4j defects
	public static String sourceDocumentsDirectory = null;	// directory to store source documents (files) of defects
	public static String sourceDocumentsIndex = null;		// directory to store source documents (files) indexes of defects
	public static String sourceResultPath = null;			// directory to store ranked suspicious source documents (files) of defects
	public static String statementDocumentsDirectory = null;  // directory to store statements (AST expressions, AST nodes, AST statements) of defects
	public static String statementDocumentsIndex = null;	// directory to store statements indexes of defects	
	public static String statementResultPath = null;		// directory to store ranked suspicious statements of defects
	public static String CommitDBPaths = null;				// path to defects4j defect info to fetch bug report URL	
	public static String XMLQueryPath  = null;				// path to store the XML queries created from bug reports
	public static String ProcessedQueryPath  = null;		// path to store the processed queries obtained from XML queries using Indri
	public static String resultsDirectory  = null;			// path to the final results directory
	public static String RerankedSuspiciousStmtsPath  = null;	// used by ranker module to re-rank results using different configurations
	public static String file_level_results_path = "file-level-fl-top50.ser"; 			// serialized file to store the computed suspicious file results
	public static String expr_level_results_path = "expression-level-fl-top10000.ser";	// serialized file to store the computed suspicious statement results
	public static String bugListFile = null; // path to the file storing all valid bug ids
	public static String flattenedBRDir = null;

	public static void setParameters(String rootDirectory){
		defects4JHome = "";
		IndriPath =  rootDirectory + "/indri-5.3";
		stopwordsFile = rootDirectory + "/stopwords"; 
		fieldsFile = rootDirectory + "/fields";
		bugCheckoutDirectory = rootDirectory + "/defects";
		sourceDocumentsDirectory = rootDirectory + "/file_docs";
		sourceDocumentsIndex = rootDirectory + "/file_indexes";
		sourceResultPath = rootDirectory + "/file_results";
		statementDocumentsDirectory = rootDirectory + "/stmt_docs";
		statementDocumentsIndex = rootDirectory + "/stmt_indexes";
		statementResultPath = rootDirectory + "/stmt_results";
		CommitDBPaths =
			  defects4JHome + "framework/projects/Chart/commit-db,"
			+ defects4JHome + "framework/projects/Cli/commit-db,"
			+ defects4JHome + "framework/projects/Closure/commit-db,"
			+ defects4JHome + "framework/projects/Codec/commit-db,"
			+ defects4JHome + "framework/projects/Collections/commit-db,"
			+ defects4JHome + "framework/projects/Compress/commit-db,"
			+ defects4JHome + "framework/projects/Csv/commit-db,"
		        + defects4JHome + "framework/projects/Gson/commit-db,"
			+ defects4JHome + "framework/projects/JacksonCore/commit-db,"
			+ defects4JHome + "framework/projects/JacksonDatabind/commit-db,"
			+ defects4JHome + "framework/projects/JacksonXml/commit-db,"
			+ defects4JHome + "framework/projects/Jsoup/commit-db,"
			+ defects4JHome + "framework/projects/JxPath/commit-db,"
			+ defects4JHome + "framework/projects/Lang/commit-db,"
			+ defects4JHome + "framework/projects/Math/commit-db,"
			+ defects4JHome + "framework/projects/Mockito/commit-db,"
			+ defects4JHome + "framework/projects/Time/commit-db";
		XMLQueryPath = rootDirectory + "/xml_queries";
		ProcessedQueryPath = rootDirectory + "/processed_queries";
		resultsDirectory = "blues_configuration_results";													
		RerankedSuspiciousStmtsPath = rootDirectory + "/" + resultsDirectory + "/" + "/blues_m";
		bugListFile = rootDirectory + "/" + "data/bug_list.txt";
		flattenedBRDir = rootDirectory + "/" + "flattened_bug_report";
	}

	public static String getDefectRepoPath(String defect) {
		return getDefectRepoPath(getProject(defect), getDefectid(defect));
	}

	public static String getDefectRepoPath(String project, String defectid) {
		return bugCheckoutDirectory + "/" + project + defectid + "buggy";
	}

	public static String getDocDir(String defect) {
		return getDocDir(getProject(defect), getDefectid(defect));
	}

	public static String getDocDir(String project, String defectid) {
		return sourceDocumentsDirectory + "/" + project + "-" + defectid + "/";
	}

	public static String getProject(String defect) {
		return defect.split("-")[0];
	}

	public static String getDefectid(String defect) {
		return defect.split("-")[1];
	}

	public static String getFilePrefix(String defect) {
		return defect;
	}

	public static void setParametersFromSettingsFile() throws IOException {
		File settings_file = new File("project.settings");
		BufferedReader br = new BufferedReader(new FileReader(settings_file));
		String line;
		while ((line = br.readLine()) != null){
			if (line.contains("root"))
				rootDirectory = line.split("=")[1].trim();
		}
		br.close();
		logger.info("Setting ROOT DIR as: " + rootDirectory);
		setParameters(rootDirectory);
	}
}
