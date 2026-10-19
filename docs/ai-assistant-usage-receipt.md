# AI assistant durable usage receipt

Schema 5 final callbacks now include a bounded `usage_manifest` (up to 16 metadata-only invocation records). The context-local collector forwards existing best-effort progress events and retains latest call states even when progress delivery fails. The existing durable business callback fence delivers the final receipt along with the generated result. No prompt, user text, or model response content is added to the manifest.

Overflow produces an empty/incomplete manifest rather than a partially chargeable list. Failed model execution still follows the existing error path. Other schemas, Cafe Host and WeCom do not gain a usage sink or manifest.

Deploy Micro's compatible final decoder **before** this API and worker. Old Micro rejects the extra final key; this is an ordered deployment, not a mixed-version final-payload negotiation. Keep quota enforcement off until Micro reservations, the receipt and actual provider/model rate keys have been verified. Roll back Deep before rolling back Micro's decoder, and drain in-flight new-format finals. Independent telemetry remains best effort; this is not supplier invoice reconciliation.
