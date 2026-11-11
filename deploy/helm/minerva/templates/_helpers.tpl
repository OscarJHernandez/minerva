{{- define "minerva.labels" -}}
app.kubernetes.io/name: minerva
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "minerva.selector" -}}
app.kubernetes.io/name: minerva
app.kubernetes.io/component: {{ .component }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
{{- end -}}
{{/* Image reference: pinned by digest when given (production), else by tag (default: chart appVersion). */}}
{{- define "minerva.image" -}}
{{- $i := .image -}}
{{- if $i.digest -}}{{ $i.repository }}@{{ $i.digest }}
{{- else if .root.Values.requireDigests -}}{{ required (printf "image digest for %s is required (requireDigests=true)" $i.repository) $i.digest }}
{{- else -}}{{ $i.repository }}:{{ $i.tag | default .root.Chart.AppVersion }}{{- end -}}
{{- end -}}
{{/* Keep pods off the nodes named in placement.*AvoidNodes (stateful = pods with volumes). */}}
{{- define "minerva.affinity" -}}
{{- $nodes := ternary .ctx.Values.placement.statefulAvoidNodes .ctx.Values.placement.statelessAvoidNodes .stateful -}}
{{- if $nodes -}}
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: NotIn
              values: {{ $nodes | toJson }}
{{- end -}}
{{- end -}}
{{- define "minerva.podSecurity" -}}
securityContext: { runAsNonRoot: true, seccompProfile: { type: RuntimeDefault } }
{{- end -}}
{{- define "minerva.containerSecurity" -}}
securityContext: { allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: { drop: [ALL] } }
{{- end -}}
