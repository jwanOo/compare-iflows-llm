import streamlit as st
import json
import requests
import zipfile
import io
import re
import html
import hashlib
from lxml import etree
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote


# =========================
# HTTP / Auth helpers
# =========================
def get_oauth_token(token_url: str, client_id: str, client_secret: str) -> str:
    """Fetch OAuth token using client credentials."""
    data = {"grant_type": "client_credentials"}
    response = requests.post(token_url, data=data, auth=(client_id, client_secret), timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "access_token" not in payload:
        raise Exception(f"OAuth response has no access_token. Keys: {list(payload.keys())}")
    return payload["access_token"]


def debug_response(response: requests.Response, label: str = "") -> bool:
    """Basic response sanity checks + helpful preview on failure."""
    content_type = (response.headers.get("content-type") or "").lower()
    if response.status_code != 200:
        st.error(f"{label} failed ({response.status_code})")
        st.code(response.text[:800])
        return False
    if "html" in content_type:
        st.error(f"{label}: got HTML (likely auth/host/redirect issue)")
        st.code(response.text[:800])
        return False
    return True


# =========================
# SAP CPI Runtime -> Id/Version
# =========================
def _text_or_none(el: Optional[etree._Element]) -> Optional[str]:
    if el is None:
        return None
    return el.text


def parse_runtime_artifacts_xml(xml_content: str) -> List[Dict[str, str]]:
    """
    Parse Atom XML feed from IntegrationRuntimeArtifacts into list of artifacts.
    Expected structure:
      atom:feed/atom:entry/m:properties/d:Id, d:Version, d:Name, d:Type, d:Status
    """
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.fromstring(xml_content.encode("utf-8"), parser)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }

    artifacts: List[Dict[str, str]] = []
    for entry in tree.findall(".//atom:entry", namespaces=ns):
        properties = entry.find("m:properties", namespaces=ns)
        if properties is None:
            continue

        artifact = {
            "Id": _text_or_none(properties.find("d:Id", namespaces=ns)) or "",
            "Version": _text_or_none(properties.find("d:Version", namespaces=ns)) or "",
            "Name": _text_or_none(properties.find("d:Name", namespaces=ns)) or "",
            "Type": _text_or_none(properties.find("d:Type", namespaces=ns)) or "",
            "Status": _text_or_none(properties.find("d:Status", namespaces=ns)) or "",
        }

        if artifact["Id"] and artifact["Name"]:
            artifacts.append(artifact)

    return artifacts


def _parse_version_tuple(v: str) -> Tuple[int, ...]:
    """
    Convert '1.2.3' -> (1,2,3) and be tolerant to unexpected formats.
    If parsing yields nothing, return (0,).
    """
    nums = re.findall(r"\d+", v or "")
    if not nums:
        return (0,)
    return tuple(int(x) for x in nums)


def find_iflow_id_and_version(host: str, iflow_name: str, access_token: str) -> Tuple[str, str]:
    """
    Search for iFlow in RuntimeArtifacts and return (Id, Version).

    Behavior:
    - If multiple deployed versions exist, prefer Status=STARTED/Started.
    - Within that, pick the highest semantic version.
    - If none are started, pick the highest version overall.
    """
    runtime_url = f"https://{host}/api/v1/IntegrationRuntimeArtifacts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/atom+xml,application/xml",
    }

    st.info(f"Searching iFlow '{iflow_name}' in RuntimeArtifacts on {host} ...")
    response = requests.get(runtime_url, headers=headers, timeout=30)
    if not debug_response(response, f"RuntimeArtifacts ({host})"):
        st.stop()

    artifacts = parse_runtime_artifacts_xml(response.text)
    st.success(f"RuntimeArtifacts loaded: {len(artifacts)} entries")

    candidates = [
        a for a in artifacts
        if a.get("Name") == iflow_name and a.get("Type") == "INTEGRATION_FLOW"
    ]

    if not candidates:
        iflows = [a for a in artifacts if a.get("Type") == "INTEGRATION_FLOW"]
        st.error(f"iFlow '{iflow_name}' not found in RuntimeArtifacts.")
        if iflows:
            st.info("Some available iFlows (first 20):")
            for a in iflows[:20]:
                st.write(f"- {a.get('Name')} (Id={a.get('Id')}, Version={a.get('Version')}, Status={a.get('Status')})")
        raise Exception(f"iFlow '{iflow_name}' not found in runtime.")

    def status_rank(status: str) -> int:
        s = (status or "").strip().upper()
        if s == "STARTED":
            return 3
        if s == "STARTING":
            return 2
        if s == "ERROR":
            return 1
        return 0

    candidates_sorted = sorted(
        candidates,
        key=lambda a: (status_rank(a.get("Status", "")), _parse_version_tuple(a.get("Version", ""))),
        reverse=True,
    )

    best = candidates_sorted[0]
    st.success(
        f"Selected latest active deployment: Id={best['Id']} Version={best['Version']} Status={best.get('Status')}"
    )

    if len(candidates_sorted) > 1:
        with st.expander("Show other deployed versions", expanded=False):
            for a in candidates_sorted[:15]:
                st.write(f"- Id={a.get('Id')} Version={a.get('Version')} Status={a.get('Status')}")

    return best["Id"], best["Version"]


# =========================
# SAP CPI DesignTime ZIP download (IMPORTANT: /$value)
# =========================
def get_designtime_iflow_zip(host: str, iflow_id: str, version: str, access_token: str) -> bytes:
    """Download iFlow ZIP from DesignTimeArtifacts $value."""
    encoded_id = quote(iflow_id, safe="")
    encoded_version = quote(version, safe="")

    designtime_url = (
        f"https://{host}/api/v1/"
        f"IntegrationDesigntimeArtifacts(Id='{encoded_id}',Version='{encoded_version}')/$value"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/octet-stream,application/zip,*/*",
    }

    st.info(f"Downloading DesignTime ZIP: {iflow_id} v{version}")
    response = requests.get(designtime_url, headers=headers, timeout=60)
    if not debug_response(response, f"DesignTimeArtifacts ZIP ({iflow_id} v{version})"):
        st.stop()

    return response.content


def download_iflow_iflw(host: str, iflow_name: str, access_token: str, display_name: str = "") -> bytes:
    """Find iFlow in RuntimeArtifacts, download its DesignTime ZIP, extract .iflw bytes."""
    iflow_id, version = find_iflow_id_and_version(host, iflow_name, access_token)
    zip_bytes = get_designtime_iflow_zip(host, iflow_id, version, access_token)

    if not zip_bytes.startswith(b"PK"):
        st.error("DesignTime download does not look like a ZIP (missing PK header).")
        st.code(zip_bytes[:800])
        raise Exception("DesignTime $value response is not a ZIP file.")

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
        names = zip_ref.namelist()
        st.info(f"ZIP entries: {len(names)}")

        for file_name in names:
            if file_name.endswith(".iflw"):
                content = zip_ref.read(file_name)
                if display_name:
                    st.caption(f"Downloaded iFlow: {display_name} (Id={iflow_id}, Version={version})")
                return content

    raise Exception("No .iflw file found inside the DesignTime ZIP.")


# =========================
# XML utilities
# =========================
def remove_xml_declaration(xml_bytes: bytes) -> str:
    xml_str = xml_bytes.decode("utf-8", errors="replace")
    return re.sub(r"^<\?xml[^>]*\?>\s*", "", xml_str)


def pretty_path(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "", path)


def _local_tag(tag: str) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else str(tag)


@st.cache_data
def build_component_label_map_from_xml(xml_str: str) -> Dict[str, str]:
    """
    Build a map: element_id -> human label like:
      'Content Modifier: Write Variables (id=CallActivity_123)'
    plus resolves BPMNShape IDs -> referenced bpmnElement labels.
    """
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    root = etree.fromstring(xml_str.encode("utf-8"), parser)

    TYPE_LABELS = {
        "contentmodifier": "Content Modifier",
        "requestreply": "Request Reply",
        "groovyscript": "Groovy Script",
        "script": "Script",
        "router": "Router",
        "splitter": "Splitter",
        "iteratingsplitter": "Iterating Splitter",
        "gather": "Gather",
        "filter": "Filter",
        "messagemapping": "Message Mapping",
        "messageMapping".lower(): "Message Mapping",
        "mapping": "Message Mapping",
    }

    BPMN_TAG_LABELS = {
        "StartEvent": "Start Event",
        "EndEvent": "End Event",
        "ExclusiveGateway": "Router (Exclusive Gateway)",
        "ParallelGateway": "Parallel Gateway",
        "ServiceTask": "Service Task",
        "CallActivity": "Call Activity",
        "IntermediateThrowEvent": "Intermediate Throw Event",
        "MessageFlow": "Message Flow",
        "Participant": "Participant",
        "SequenceFlow": "Sequence Flow",
    }

    def first_attr(e, keys: List[str]) -> str:
        for k in keys:
            v = e.attrib.get(k)
            if v:
                return v.strip()
        return ""

    def friendly_type(e) -> str:
        type_hint = first_attr(
            e,
            [
                "{http:///com.sap.ifl.model/Ifl.xsd}type",
                "activityType",
                "ComponentType",
                "componentType",
                "type",
            ],
        )
        if type_hint:
            key = type_hint.replace(" ", "").replace("-", "").lower()
            return TYPE_LABELS.get(key, type_hint)
        return BPMN_TAG_LABELS.get(_local_tag(e.tag), _local_tag(e.tag))

    def display_name(e) -> str:
        name = first_attr(e, ["name", "Name"])
        if name:
            return name
        alt = first_attr(e, ["mappingname", "scriptFunction"])
        return alt if alt else "<unnamed>"

    label_map: Dict[str, str] = {}

    for e in root.iter():
        eid = e.attrib.get("id")
        if not eid:
            continue
        t = friendly_type(e)
        n = display_name(e)
        label_map[eid] = f"{t}: {n} (id={eid})"

    for e in root.iter():
        if _local_tag(e.tag) != "BPMNShape":
            continue
        shape_id = e.attrib.get("id")
        ref = (e.attrib.get("bpmnElement") or "").replace("BPMNShape_", "")
        if shape_id and ref:
            label_map[shape_id] = label_map.get(ref, f"BPMNShape: <unknown> (id={shape_id})")

    for eid, lbl in list(label_map.items()):
        if not eid.startswith("BPMNShape_"):
            label_map[f"BPMNShape_{eid}"] = lbl

    return label_map


def label_for_id(eid: Optional[str], label_map: Optional[Dict[str, str]]) -> str:
    if not eid:
        return "<none>"
    if not label_map:
        return eid
    return label_map.get(eid, eid)


# =========================
# XML comparison logic
# =========================
def compare_elements(
    e1,
    e2,
    path: str = "/",
    api1_name: str = "API 1",
    api2_name: str = "API 2",
    label_map1: Optional[Dict[str, str]] = None,
    label_map2: Optional[Dict[str, str]] = None,
) -> List[str]:
    tag1 = _local_tag(e1.tag)
    tag2 = _local_tag(e2.tag)

    id1 = e1.attrib.get("id")
    id2 = e2.attrib.get("id")

    any_id = id1 or id2
    extra = f" ({label_for_id(any_id, {**(label_map1 or {}), **(label_map2 or {})})})" if any_id else ""

    differences: List[str] = []

    if tag1 == "BPMNEdge" and tag2 == "BPMNEdge":
        src1 = e1.attrib.get("sourceElement", "").replace("BPMNShape_", "")
        tgt1 = e1.attrib.get("targetElement", "").replace("BPMNShape_", "")
        src2 = e2.attrib.get("sourceElement", "").replace("BPMNShape_", "")
        tgt2 = e2.attrib.get("targetElement", "").replace("BPMNShape_", "")

        src_name1 = label_for_id(src1, label_map1)
        tgt_name1 = label_for_id(tgt1, label_map1)
        src_name2 = label_for_id(src2, label_map2)
        tgt_name2 = label_for_id(tgt2, label_map2)

        if src1 != src2 or tgt1 != tgt2:
            differences.append(
                f"{pretty_path(path)} (BPMNEdge {any_id}):\n"
                f"  {api1_name}: source={src1} -> {src_name1}, target={tgt1} -> {tgt_name1}\n"
                f"  {api2_name}: source={src2} -> {src_name2}, target={tgt2} -> {tgt_name2}"
            )
        return differences

    if tag1 == "BPMNShape" and tag2 == "BPMNShape":
        ref_id1 = e1.attrib.get("bpmnElement", "").replace("BPMNShape_", "")
        ref_id2 = e2.attrib.get("bpmnElement", "").replace("BPMNShape_", "")
        name1 = label_for_id(ref_id1, label_map1)
        name2 = label_for_id(ref_id2, label_map2)

        if ref_id1 != ref_id2:
            differences.append(
                f"{pretty_path(path)} (BPMNShape {any_id}):\n"
                f"  {api1_name}: bpmnElement={ref_id1} -> {name1}\n"
                f"  {api2_name}: bpmnElement={ref_id2} -> {name2}"
            )
        return differences

    if id1 and id2 and id1 == id2:
        if e1.attrib != e2.attrib:
            differences.append(
                f"{pretty_path(path)}{extra}: Attributes differ\n"
                f"  {api1_name}: {dict(e1.attrib)}\n"
                f"  {api2_name}: {dict(e2.attrib)}"
            )

        text1 = (e1.text or "").strip()
        text2 = (e2.text or "").strip()
        if text1 != text2:
            differences.append(
                f"{pretty_path(path)}{extra}: Text differs\n"
                f"  {api1_name}: '{text1}'\n"
                f"  {api2_name}: '{text2}'"
            )

        children1 = list(e1)
        children2 = list(e2)

        id_map1 = {c.attrib["id"]: c for c in children1 if "id" in c.attrib}
        id_map2 = {c.attrib["id"]: c for c in children2 if "id" in c.attrib}

        for cid in set(id_map1.keys()).union(id_map2.keys()):
            if cid not in id_map1:
                differences.append(
                    f"{pretty_path(path)}: Present in {api2_name} but missing in {api1_name} -> {label_for_id(cid, label_map2)}"
                )
            elif cid not in id_map2:
                differences.append(
                    f"{pretty_path(path)}: Present in {api1_name} but missing in {api2_name} -> {label_for_id(cid, label_map1)}"
                )
            else:
                differences.extend(
                    compare_elements(
                        id_map1[cid],
                        id_map2[cid],
                        f"{path}/{_local_tag(id_map1[cid].tag)}[id={cid}]",
                        api1_name,
                        api2_name,
                        label_map1,
                        label_map2,
                    )
                )

        children1_noid = [c for c in children1 if "id" not in c.attrib]
        children2_noid = [c for c in children2 if "id" not in c.attrib]

        if len(children1_noid) != len(children2_noid):
            differences.append(
                f"{pretty_path(path)}{extra}: Number of child elements without id differ - "
                f"{api1_name}: {len(children1_noid)}, {api2_name}: {len(children2_noid)}"
            )
        else:
            for i, (c1, c2) in enumerate(zip(children1_noid, children2_noid)):
                child_path = f"{path}/{_local_tag(c1.tag)}[{i}]"
                differences.extend(compare_elements(c1, c2, child_path, api1_name, api2_name, label_map1, label_map2))

        return differences

    if tag1 != tag2:
        differences.append(f"{pretty_path(path)}{extra}: Tag differs - {api1_name}: {tag1}, {api2_name}: {tag2}")

    if e1.attrib != e2.attrib:
        differences.append(
            f"{pretty_path(path)}{extra}: Attributes differ\n"
            f"  {api1_name}: {dict(e1.attrib)}\n"
            f"  {api2_name}: {dict(e2.attrib)}"
        )

    text1 = (e1.text or "").strip()
    text2 = (e2.text or "").strip()
    if text1 != text2:
        differences.append(
            f"{pretty_path(path)}{extra}: Text differs\n"
            f"  {api1_name}: '{text1}'\n"
            f"  {api2_name}: '{text2}'"
        )

    children1 = list(e1)
    children2 = list(e2)
    if len(children1) != len(children2):
        differences.append(
            f"{pretty_path(path)}{extra}: Number of child elements differ - {api1_name}: {len(children1)}, {api2_name}: {len(children2)}"
        )
    else:
        for i, (c1, c2) in enumerate(zip(children1, children2)):
            child_path = f"{path}/{_local_tag(c1.tag)}[{i}]"
            differences.extend(compare_elements(c1, c2, child_path, api1_name, api2_name, label_map1, label_map2))

    return differences


def run_detailed_xml_comparison(xml1_str: str, xml2_str: str, api1_name: str, api2_name: str) -> List[str]:
    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree1 = etree.fromstring(xml1_str.encode("utf-8"), parser)
    tree2 = etree.fromstring(xml2_str.encode("utf-8"), parser)

    label_map1 = build_component_label_map_from_xml(xml1_str)
    label_map2 = build_component_label_map_from_xml(xml2_str)

    return compare_elements(tree1, tree2, "/", api1_name, api2_name, label_map1, label_map2)


def annotate_ids_with_names(differences: List[str], label_map: Dict[str, str]) -> List[str]:
    def annotate_line(line: str) -> str:
        def repl(match):
            eid = match.group(0)
            return label_map.get(eid, eid)

        return re.sub(r"\b[A-Za-z_]+_\d+\b", repl, line)

    return [annotate_line(line) for line in differences]


# =========================
# LLM
# =========================
def _normalize_openai_url(endpoint: str) -> str:
    e = endpoint.rstrip("/")
    if e.endswith("/chat/completions"):
        return e
    if e.endswith("/v1"):
        return f"{e}/chat/completions"
    return e


def call_llm(prompt: str, llm_config: Dict[str, str]) -> str:
    provider = (llm_config.get("provider") or "").lower().strip()
    endpoint = (llm_config.get("endpoint") or "").strip()
    api_key = (llm_config.get("api_key") or "").strip()
    model = (llm_config.get("model") or "").strip()

    if not provider or not endpoint or not api_key:
        return "LLM config incomplete: provider/endpoint/api_key required."

    timeout_s = 90

    if provider == "gemini":
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        resp = requests.post(endpoint, headers=headers, json=data, timeout=timeout_s)
        if resp.status_code != 200:
            return f"Gemini error: {resp.status_code} {resp.text[:400]}"

        result = resp.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return f"Gemini: unexpected response shape. Keys: {list(result.keys())}"

    if provider == "openai":
        url = _normalize_openai_url(endpoint)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        body = {
            "model": model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        if resp.status_code != 200:
            return f"OpenAI error: {resp.status_code} {resp.text[:400]}"

        result = resp.json()
        try:
            return result["choices"][0]["message"]["content"]
        except Exception:
            return f"OpenAI: unexpected response shape. Keys: {list(result.keys())}"

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": model or "claude-3-5-sonnet-20240620",
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }

        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout_s)
        if resp.status_code != 200:
            return f"Anthropic error: {resp.status_code} {resp.text[:400]}"

        result = resp.json()
        try:
            return result["content"][0]["text"]
        except Exception:
            return f"Anthropic: unexpected response shape. Keys: {list(result.keys())}"

    return f"Unsupported LLM provider: {provider}"


def clean_markdown(text: str) -> str:
    return re.sub(r"\*\*(.*?)\*\*|\*(.*?)\*", r"\1\2", text).replace("\n\n", "\n")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Compare iFlows with LLMs", page_icon="🔄", layout="wide")

st.title("🔄 Compare iFlows with LLMs")
st.caption("Compare two deployed iFlows and generate a summary using your selected LLM.")

# =========================
# Config file loader (sidebar) -> populate session_state and rerun
# =========================
st.sidebar.subheader("⚙️ Configuration")
config_file = st.sidebar.file_uploader("Load config_file.json", type="json")

def apply_config_to_state(cfg: Dict[str, Dict[str, str]]) -> None:
    a1 = cfg.get("api1", {}) or {}
    a2 = cfg.get("api2", {}) or {}
    # Environment 1
    st.session_state["api1_host"] = a1.get("host_url", "")
    st.session_state["api1_iflow"] = a1.get("name", "")
    st.session_state["api1_name"] = a1.get("name", "")
    st.session_state["api1_token_url"] = a1.get("oauth_token_url", "")
    st.session_state["api1_client_id"] = a1.get("client_id", "")
    st.session_state["api1_client_secret"] = a1.get("client_secret", "")
    # Environment 2
    st.session_state["api2_host"] = a2.get("host_url", "")
    st.session_state["api2_iflow"] = a2.get("name", "")
    st.session_state["api2_name"] = a2.get("name", "")
    st.session_state["api2_token_url"] = a2.get("oauth_token_url", "")
    st.session_state["api2_client_id"] = a2.get("client_id", "")
    st.session_state["api2_client_secret"] = a2.get("client_secret", "")

if config_file is not None:
    file_bytes = config_file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("_config_hash") != file_hash:
        try:
            cfg = json.loads(file_bytes.decode("utf-8"))
            apply_config_to_state(cfg)
            st.session_state["_config_hash"] = file_hash
            st.sidebar.success("Config loaded and applied to form.")
            st.rerun()  # reflect new values in widgets immediately [web:262]
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")

# Helper to initialize state-backed defaults before rendering widgets
def ss_default(key: str, default: str = "") -> str:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# =========================
# LLM UI (outside form so it updates immediately)
# =========================
st.subheader("🤖 LLM Summary")

LLM_DEFAULT_ENDPOINTS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}

MODEL_PRESETS = {
    "gemini": ["gemini-pro", "gemini-1.5-pro", "gemini-2.0-flash"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"],
}

def default_endpoint_for(provider: str, model: str) -> str:
    provider = (provider or "").lower().strip()
    model = (model or "").strip()
    if provider == "openai":
        return LLM_DEFAULT_ENDPOINTS["openai"]
    if provider == "anthropic":
        return LLM_DEFAULT_ENDPOINTS["anthropic"]
    if provider == "gemini":
        if model:
            return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        return LLM_DEFAULT_ENDPOINTS["gemini"]
    return ""

llm_provider = st.selectbox("LLM Provider", ["gemini", "openai", "anthropic"], key="llm_provider")

preset_models = MODEL_PRESETS.get(llm_provider, [])
default_model = preset_models[0] if preset_models else ""

model_state_key = f"llm_model_{llm_provider}"
if model_state_key not in st.session_state:
    st.session_state[model_state_key] = default_model

llm_model = st.selectbox("Model", options=preset_models, key=model_state_key)

override_endpoint = st.checkbox("Override API endpoint", value=False, key="override_endpoint")

auto_endpoint = default_endpoint_for(llm_provider, llm_model)

if "llm_endpoint" not in st.session_state:
    st.session_state["llm_endpoint"] = auto_endpoint

if not override_endpoint:
    prev_provider = st.session_state.get("_prev_provider")
    prev_model = st.session_state.get("_prev_model")
    provider_changed = prev_provider != llm_provider
    model_changed = prev_model != llm_model
    if provider_changed or (llm_provider == "gemini" and model_changed):
        st.session_state["llm_endpoint"] = auto_endpoint

st.session_state["_prev_provider"] = llm_provider
st.session_state["_prev_model"] = llm_model

st.text_input("API Endpoint (auto-filled)", key="llm_endpoint")
llm_api_key = st.text_input("API Key", type="password", key="llm_api_key")

llm_config: Dict[str, str] = {
    "provider": llm_provider,
    "endpoint": st.session_state["llm_endpoint"],
    "api_key": llm_api_key,
}
if llm_provider != "gemini":
    llm_config["model"] = llm_model

st.divider()

# =========================
# Main form (env + auth; values come from session_state)
# =========================
with st.form("compare_form", border=True):
    st.subheader("🌐 Environments")
    env1, env2 = st.columns(2)

    with env1:
        st.markdown("**1️⃣ Environment 1**")
        ss_default("api1_host", "Enter Your CPI Host")
        ss_default("api1_iflow", "Enter Source iFlow Name")
        ss_default("api1_name", "Source iFlow")
        st.text_input("Host", key="api1_host")
        st.text_input("iFlow Name", key="api1_iflow")
        st.text_input("Display Name", key="api1_name")

    with env2:
        st.markdown("**2️⃣ Environment 2**")
        ss_default("api2_host", "Enter Your CPI Host")
        ss_default("api2_iflow", "Submit Target iFlow Name")
        ss_default("api2_name", "Target iFlow")
        st.text_input("Host", key="api2_host")
        st.text_input("iFlow Name", key="api2_iflow")
        st.text_input("Display Name", key="api2_name")

    st.divider()
    st.subheader("🔐 Authentication")
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Environment 1**")
        ss_default("api1_token_url", "Enter your OAuth Token URL")
        ss_default("api1_client_id", "")
        ss_default("api1_client_secret", "")
        st.text_input("OAuth Token URL", key="api1_token_url")
        st.text_input("Client ID", type="password", key="api1_client_id")
        st.text_input("Client Secret", type="password", key="api1_client_secret")

    with a2:
        st.markdown("**Environment 2**")
        ss_default("api2_token_url", "Enter Your OAuth Token URL")
        ss_default("api2_client_id", "")
        ss_default("api2_client_secret", "")
        st.text_input("OAuth Token URL", key="api2_token_url")
        st.text_input("Client ID", type="password", key="api2_client_id")
        st.text_input("Client Secret", type="password", key="api2_client_secret")

    st.divider()
    submitted = st.form_submit_button("▶️ Compare iFlows", use_container_width=True)


if submitted:
    required = [
        st.session_state.get("api1_host"), st.session_state.get("api1_iflow"),
        st.session_state.get("api1_token_url"), st.session_state.get("api1_client_id"), st.session_state.get("api1_client_secret"),
        st.session_state.get("api2_host"), st.session_state.get("api2_iflow"),
        st.session_state.get("api2_token_url"), st.session_state.get("api2_client_id"), st.session_state.get("api2_client_secret"),
        llm_api_key,
    ]

    if not all(required):
        st.error("Please fill all required fields.")
    else:
        try:
            api1_host = st.session_state["api1_host"]
            api1_iflow = st.session_state["api1_iflow"]
            api1_name = st.session_state["api1_name"]
            api1_token_url = st.session_state["api1_token_url"]
            api1_client_id = st.session_state["api1_client_id"]
            api1_client_secret = st.session_state["api1_client_secret"]

            api2_host = st.session_state["api2_host"]
            api2_iflow = st.session_state["api2_iflow"]
            api2_name = st.session_state["api2_name"]
            api2_token_url = st.session_state["api2_token_url"]
            api2_client_id = st.session_state["api2_client_id"]
            api2_client_secret = st.session_state["api2_client_secret"]

            with st.spinner("🔐 Authenticating..."):
                token1 = get_oauth_token(api1_token_url, api1_client_id, api1_client_secret)
                token2 = get_oauth_token(api2_token_url, api2_client_id, api2_client_secret)

            with st.spinner("⬇️ Downloading iFlows..."):
                xml1_bytes = download_iflow_iflw(api1_host, api1_iflow, token1, api1_name)
                xml2_bytes = download_iflow_iflw(api2_host, api2_iflow, token2, api2_name)

            xml1 = remove_xml_declaration(xml1_bytes)
            xml2 = remove_xml_declaration(xml2_bytes)

            with st.spinner("🔎 Comparing XML..."):
                differences = run_detailed_xml_comparison(xml1, xml2, api1_name, api2_name)

                label_map1 = build_component_label_map_from_xml(xml1)
                label_map2 = build_component_label_map_from_xml(xml2)
                combined_label_map = {**label_map1, **label_map2}

                differences_annotated = annotate_ids_with_names(differences, combined_label_map)

            if not differences_annotated:
                st.success("✅ There are no differences in the iFlows.")
                st.stop()

            differences_str = "\n\n".join(differences_annotated)

            st.subheader("📋 Detailed differences (human-readable)")
            st.text_area("Differences", differences_str, height=320, key="diff_text")
            st.download_button(
                "⬇️ Download raw differences",
                differences_str,
                file_name="iflow_detailed_differences.txt",
            )

            with st.spinner("🧠 Generating LLM summary..."):
                prompt = (
                    f"Analyze these SAP CPI iFlow differences between '{api1_name}' and '{api2_name}':\n\n"
                    f"{differences_str}\n\n"
                    "Provide a clear bullet-point summary of:\n"
                    "- Configuration changes (properties, scripts, mappings)\n"
                    "- Process flow changes (activities added/removed/changed)\n"
                    "- Critical structural differences\n"
                    "- Value modifications in Call Activities\n\n"
                    "Ignore: graphical layout, coordinates, BPMNEdge/SequenceFlow positioning.\n"
                    "Focus on: functional differences that impact runtime behavior."
                )
                summary = call_llm(prompt, llm_config)

            summary = html.unescape(summary)
            summary = clean_markdown(summary)

            st.subheader("🧠 LLM summary")
            st.text_area("Summary", summary, height=320, key="summary_text")
            st.download_button("⬇️ Download summary", summary, file_name="iflow_llm_summary.txt")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
