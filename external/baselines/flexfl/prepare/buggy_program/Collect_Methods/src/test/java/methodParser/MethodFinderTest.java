package methodParser;

import static groundTruthBuilder.MethodFinder.extractMethodSignature;
import static org.junit.Assert.*;

public class MethodFinderTest {

    @org.junit.Test
    public void testExtractMethodSignature() {
        String method = "@SuppressWarnings(\"unchecked\") // returned value will have type T because it is fixed by clazz\n    public static <T> T createValue(final String str, final Class<T> clazz) throws ParseException\n    {\n        if (PatternOptionBuilder.STRING_VALUE == clazz)\n        {\n            return (T) str;\n        }\n        else if (PatternOptionBuilder.OBJECT_VALUE == clazz)\n        {\n            return (T) createObject(str);\n        }\n        else if (PatternOptionBuilder.NUMBER_VALUE == clazz)\n        {\n            return (T) createNumber(str);\n        }\n        else if (PatternOptionBuilder.DATE_VALUE == clazz)\n        {\n            return (T) createDate(str);\n        }\n        else if (PatternOptionBuilder.CLASS_VALUE == clazz)\n        {\n            return (T) createClass(str);\n        }\n        else if (PatternOptionBuilder.FILE_VALUE == clazz)\n        {\n            return (T) createFile(str);\n        }\n        else if (PatternOptionBuilder.EXISTING_FILE_VALUE == clazz)\n        {\n            return (T) openFile(str);\n        }\n        else if (PatternOptionBuilder.FILES_VALUE == clazz)\n        {\n            return (T) createFiles(str);\n        }\n        else if (PatternOptionBuilder.URL_VALUE == clazz)\n        {\n            return (T) createURL(str);\n        }\n        else\n        {\n            return null;\n        }\n    }";
        String signature = extractMethodSignature(method);
        assertEquals(signature, "createValue(String,Class<T>)");
    }
}