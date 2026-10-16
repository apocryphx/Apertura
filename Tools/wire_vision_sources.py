#!/usr/bin/env python3
"""Register new aptransformer/ES*.mm sources in Apertura.xcodeproj for every target that
compiles the ES core directly (mirrors an existing core file's entries: file reference,
AperturaCore-src group membership, and one PBXBuildFile per Sources phase). Idempotent."""
import hashlib, re, sys, os

PBX = os.path.join(os.path.dirname(__file__), "..", "Apertura.xcodeproj", "project.pbxproj")
MIRROR = "ESDecodeAttn.mm"   # default: aptransformer core files
SUBDIR = "aptransformer"

def oid(seed):
    return hashlib.md5(seed.encode()).hexdigest()[:24].upper()

def main(names):
    global MIRROR, SUBDIR
    if names and names[0].startswith("--mirror="):   # --mirror=APGoogleSession.m --dir=AperturaKit
        MIRROR = names[0].split("=", 1)[1]; names = names[1:]
    if names and names[0].startswith("--dir="):
        SUBDIR = names[0].split("=", 1)[1]; names = names[1:]
    s = open(PBX).read()
    m = re.search(r"\t\t([0-9A-F]{24}) /\* %s \*/ = \{isa = PBXFileReference;" % re.escape(MIRROR), s)
    mirror_ref = m.group(1)
    phases = re.findall(r"\t\t\t\t([0-9A-F]{24}) /\* %s in Sources \*/,\n" % re.escape(MIRROR), s)
    assert phases, "mirror file not in any Sources phase"
    for name in names:
        if f"/* {name} */ = {{isa = PBXFileReference" in s:
            print("already wired:", name); continue
        ref = oid("ref:" + name)
        s = s.replace(
            f"\t\t{mirror_ref} /* {MIRROR} */ = {{isa = PBXFileReference;",
            f"\t\t{ref} /* {name} */ = {{isa = PBXFileReference; includeInIndex = 1; lastKnownFileType = {'sourcecode.c.objc' if name.endswith('.m') else 'sourcecode.cpp.objcpp'}; "
            f"name = {name}; path = {SUBDIR}/{name}; sourceTree = \"<group>\"; }};\n"
            f"\t\t{mirror_ref} /* {MIRROR} */ = {{isa = PBXFileReference;", 1)
        # group membership: after the mirror's child line in AperturaCore-src
        s = s.replace(f"\t\t\t\t{mirror_ref} /* {MIRROR} */,\n",
                      f"\t\t\t\t{mirror_ref} /* {MIRROR} */,\n\t\t\t\t{ref} /* {name} */,\n", 1)
        for i, bf in enumerate(phases):
            nbf = oid(f"bf{i}:" + name)
            s = s.replace(f"\t\t{bf} /* {MIRROR} in Sources */ = {{isa = PBXBuildFile; fileRef = {mirror_ref} /* {MIRROR} */; }};\n",
                          f"\t\t{bf} /* {MIRROR} in Sources */ = {{isa = PBXBuildFile; fileRef = {mirror_ref} /* {MIRROR} */; }};\n"
                          f"\t\t{nbf} /* {name} in Sources */ = {{isa = PBXBuildFile; fileRef = {ref} /* {name} */; }};\n", 1)
            s = s.replace(f"\t\t\t\t{bf} /* {MIRROR} in Sources */,\n",
                          f"\t\t\t\t{bf} /* {MIRROR} in Sources */,\n\t\t\t\t{nbf} /* {name} in Sources */,\n", 1)
        print("wired:", name, "->", len(phases), "targets")
    open(PBX, "w").write(s)

if __name__ == "__main__":
    main(sys.argv[1:])
