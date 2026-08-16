#!/bin/bash

# ============================================================
# CAMD local environment setup
# ============================================================

CAMD_ROOT="/Users/bao/Documents/GitHub/CAMD"

# Java 11
export JAVA_HOME="/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"

# Perl local modules
export PERL5LIB="$HOME/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}"

# Defects4J
export DEFECTS4J_HOME="$CAMD_ROOT/external/defects4j"

# Hugging Face local cache
export HF_HOME="$CAMD_ROOT/hf_home"

# Force Hugging Face to use local cache only
export HF_HUB_OFFLINE=1

# Executable paths
export PATH="$JAVA_HOME/bin:$HOME/perl5/bin:$DEFECTS4J_HOME/framework/bin:$PATH"


echo "============================================================"
echo "CAMD environment loaded"
echo "============================================================"

echo
echo "CAMD_ROOT:"
echo "  $CAMD_ROOT"

echo
echo "JAVA_HOME:"
echo "  $JAVA_HOME"

echo
echo "Java:"
java -version 2>&1 | head -n 1

echo
echo "Perl:"
perl -MString::Interpolate -e 'print "  String::Interpolate OK\n"' 2>/dev/null \
    || echo "  WARNING: String::Interpolate not found"

echo
echo "Defects4J:"
if command -v defects4j >/dev/null 2>&1; then
    echo "  $(command -v defects4j)"
else
    echo "  WARNING: defects4j not found"
fi

echo
echo "HF_HOME:"
echo "  $HF_HOME"

echo
echo "HF_HUB_OFFLINE:"
echo "  $HF_HUB_OFFLINE"

echo
echo "Python:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "  venv: $VIRTUAL_ENV"
    echo "  python: $(command -v python)"
else
    echo "  WARNING: Python venv is not activated"
    echo "  python: $(command -v python)"
fi

echo
echo "============================================================"