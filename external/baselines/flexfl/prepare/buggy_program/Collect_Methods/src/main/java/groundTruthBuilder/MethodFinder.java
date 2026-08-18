package groundTruthBuilder;

import org.apache.log4j.Logger;
import org.eclipse.jdt.core.dom.*;

import java.util.*;

public class MethodFinder {

    protected static Logger logger = Logger.getLogger(MethodFinder.class);
    private Set<String> signatures;
    private String fileContent;
    private CompilationUnit cu;
    private String packageName;

    public MethodFinder(String fileContent){
        this.fileContent = fileContent;
        this.signatures = new HashSet<>();
        this.cu = null;
    }

    public static String extractMethodSignature(String method) {
        method = "public class Temp {\n" + method + "\n}";
        ASTParser parser = ASTParser.newParser(4);
        parser.setKind(8);
        parser.setSource(method.toCharArray());
        CompilationUnit cu = (CompilationUnit) parser.createAST(null);
        List<String> methodSignatures = new ArrayList<>();
        cu.accept(new ASTVisitor() {
            @Override
            public boolean visit(MethodDeclaration methodDeclaration) {
                StringBuilder idMethod = new StringBuilder(methodDeclaration.getName().getFullyQualifiedName());
                // cannot be instance
                // TypeDeclaration parentOfMethod = (TypeDeclaration) methodDeclaration.getParent();
                // if (parentOfMethod.isInterface())
                //        return;
                // cannot be abstract
                // if (Modifier.isAbstract(methodDeclaration.getModifiers())) {
                //    return;
                // }
                List<SingleVariableDeclaration> listOfParameters = methodDeclaration.parameters();
                idMethod.append("(");
                List<String> paramTypes = new ArrayList<>();
                for (SingleVariableDeclaration p : listOfParameters) {
                    paramTypes.add(p.getType().toString());
                }
                idMethod.append(String.join(",", paramTypes)).append(")");
                methodSignatures.add(idMethod.toString());
                // 不用继续往下parse了
                return true;
            }
        });

        assert methodSignatures.size() > 0;
        return methodSignatures.get(0);
    }

    public CompilationUnit parseSourceCode() {
        char[] fileContentAsChar = this.fileContent.toCharArray();
        ASTParser parser = ASTParser.newParser(4);
        parser.setKind(8);
        parser.setSource(fileContentAsChar);
        cu = (CompilationUnit) parser.createAST(null);
        return cu;
    }

    public void exploreSourceCode(CompilationUnit compilationUnitSourceCode) {
        if (compilationUnitSourceCode.getPackage() != null)
            this.packageName = compilationUnitSourceCode.getPackage().getName().toString();
        List<ASTNode> declaredTypes = compilationUnitSourceCode.types();
        for (ASTNode currentDeclaredType : declaredTypes) {
            if (currentDeclaredType.getNodeType() == 55) {
                TypeDeclaration typeDeclaration = (TypeDeclaration) currentDeclaredType;
                exploreClassContents((TypeDeclaration) currentDeclaredType, "");
            }
        }
    }

    private void exploreClassContents(TypeDeclaration classNode, String prefixClass) {
        List<ASTNode> bodyDeclarations = classNode.bodyDeclarations();
        SimpleName className = classNode.getName();
        String fullClassName = prefixClass + className + ".";
        for (ASTNode bodyDeclaration : bodyDeclarations) {
            if (bodyDeclaration.getNodeType() == 31)
                exploreMethodContents((MethodDeclaration) bodyDeclaration, fullClassName);
            if (bodyDeclaration.getNodeType() == 55)
                exploreClassContents((TypeDeclaration) bodyDeclaration, fullClassName);
        }
    }

    public static Integer[] getLineNumbers(MethodDeclaration methodDecl, CompilationUnit cu) {
        int startLineNo = cu.getLineNumber(methodDecl.getStartPosition());
        int nodeLength = methodDecl.getLength();
        int endLineNo = cu.getLineNumber(methodDecl.getStartPosition() + nodeLength);
        return new Integer[]{startLineNo, endLineNo};
    }

    private void exploreMethodContents(MethodDeclaration methodDeclaration, String fullClassName) {
        String currentMethodName = methodDeclaration.getName().getFullyQualifiedName();
        String idMethod = fullClassName + currentMethodName;
        TypeDeclaration parentOfMethod = (TypeDeclaration) methodDeclaration.getParent();
        if (parentOfMethod.isInterface())
            return;
        if (Modifier.isAbstract(methodDeclaration.getModifiers())) {
            logger.debug("ID=" + idMethod);
            logger.debug("Abstract method (ignored) ID=" + idMethod);
            return;
        }
        List<SingleVariableDeclaration> listOfParameters = methodDeclaration.parameters();
        idMethod = idMethod + "\t" + listOfParameters.size() + "\t";
        for (SingleVariableDeclaration p : listOfParameters)
            idMethod = idMethod + p.getType() + "\t";
        String idMethodWithPackageSeparator = this.packageName + "$" + idMethod;
        idMethod = this.packageName + "." + idMethod;
        String signature = convertMethodIDToFinalFormat(idMethod);
        assert !this.signatures.contains(signature);

        Integer[] lineNos = getLineNumbers(methodDeclaration, cu);
        signature = signature + "." + lineNos[0] + "." + lineNos[1];

        this.signatures.add(signature);
    }

    private static String convertMethodIDToFinalFormat(String idMethod) {
        String[] splittedBuf = idMethod.split("\t");
        String methodNameFullPath = splittedBuf[0];
        String methodNameFullPathFinal = methodNameFullPath + "(";
        int numberOfParameters = Integer.parseInt(splittedBuf[1]);
        if (numberOfParameters != 0) {
            for (int indexParameter = 0; indexParameter < numberOfParameters - 1; indexParameter++)
                methodNameFullPathFinal = methodNameFullPathFinal + splittedBuf[indexParameter + 2] + ",";
            methodNameFullPathFinal = methodNameFullPathFinal + splittedBuf[numberOfParameters + 1];
        }
        methodNameFullPathFinal = methodNameFullPathFinal + ")";
        return methodNameFullPathFinal;
    }

    public String findMethod(String targetShortSig) {
        CompilationUnit cu = parseSourceCode();
        exploreSourceCode(cu);
        List<String> findSigs = new ArrayList<>();
        for (String sig : signatures){
            String[] items = sig.split("\\.");
            // method(xx, xx);
            String shortSig = items[items.length-3];
            if (shortSig.equals(targetShortSig)) {
                findSigs.add(sig);
            }
        }
        assert findSigs.size() == 1;
        return findSigs.get(0);
    }
}
