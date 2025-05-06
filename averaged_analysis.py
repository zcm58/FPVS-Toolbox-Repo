# advanced_analysis.py
import tkinter as tk
from tkinter import filedialog, messagebox as tk_messagebox
import customtkinter as ctk
import os
import re  # For PID extraction later
import threading  # For background processing
import time  # For simulating work in placeholder
from typing import List, Dict, Any, Optional, Union  # For type hints

# Attempt to import shared constants, with fallbacks
try:
    from config import (PAD_X, PAD_Y, CORNER_RADIUS, ENTRY_WIDTH,
                        BUTTON_WIDTH as CONFIG_BUTTON_WIDTH,
                        ADV_ENTRY_WIDTH as CONFIG_ADV_ENTRY_WIDTH,
                        ADV_LABEL_ID_ENTRY_WIDTH as CONFIG_ADV_LABEL_ID_ENTRY_WIDTH,
                        ADV_ID_ENTRY_WIDTH as CONFIG_ADV_ID_ENTRY_WIDTH)

    BUTTON_WIDTH = CONFIG_BUTTON_WIDTH
    ADV_ENTRY_WIDTH = CONFIG_ADV_ENTRY_WIDTH
    ADV_LABEL_ID_ENTRY_WIDTH = CONFIG_ADV_LABEL_ID_ENTRY_WIDTH
    ADV_ID_ENTRY_WIDTH = CONFIG_ADV_ID_ENTRY_WIDTH
except ImportError:
    PAD_X = 5
    PAD_Y = 5
    CORNER_RADIUS = 8
    ENTRY_WIDTH = 120
    BUTTON_WIDTH = 180
    ADV_ENTRY_WIDTH = ENTRY_WIDTH
    ADV_LABEL_ID_ENTRY_WIDTH = int(ENTRY_WIDTH * 1.5)
    ADV_ID_ENTRY_WIDTH = int(ENTRY_WIDTH * 0.5)


# Helper to create themed Listbox
def create_themed_listbox(parent: ctk.CTkBaseClass, **kwargs) -> tk.Listbox:
    """Creates a tk.Listbox with theming consistent with CustomTkinter."""
    current_appearance = ctk.get_appearance_mode()
    listbox = tk.Listbox(
        parent,
        background=ctk.ThemeManager.theme["CTkFrame"]["fg_color"][current_appearance],
        fg=ctk.ThemeManager.theme["CTkLabel"]["text_color"][current_appearance],
        highlightbackground=ctk.ThemeManager.theme["CTkFrame"]["border_color"][current_appearance],
        highlightcolor=ctk.ThemeManager.theme["CTkButton"]["fg_color"][current_appearance],
        selectbackground=ctk.ThemeManager.theme["CTkButton"]["hover_color"][current_appearance],
        selectforeground=ctk.ThemeManager.theme["CTkButton"]["text_color"][current_appearance],
        borderwidth=0, highlightthickness=1, activestyle='none',
        **kwargs
    )
    return listbox


class AdvancedAnalysisWindow(ctk.CTkToplevel):
    """Toplevel window for advanced averaging analysis."""

    def __init__(self, master: ctk.CTkBaseClass):
        super().__init__(master)
        self.master_app = master

        self.title("Advanced Averaging Analysis")
        self.geometry("1050x850")
        self.minsize(950, 750)

        self.source_eeg_files: List[str] = []
        self.defined_groups: List[Dict[str, Any]] = []
        self.selected_group_index: Optional[int] = None

        self.processing_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()  # For graceful thread termination

        self._build_ui()
        self.log("Advanced Averaging Analysis window initialized.")
        self._check_main_app_params()

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._center_window)
        self._update_start_processing_button_state()

    def _check_main_app_params(self) -> None:
        """Checks if main app parameters are available."""
        if hasattr(self.master_app, 'validated_params') and self.master_app.validated_params:
            self.log("Preprocessing parameters will be based on the main app's last validated settings.")
        else:
            self.log("Warning: Main app's parameters not found. Default preprocessing might be used or fail.")
        if not hasattr(self.master_app, 'save_folder_path') or not self.master_app.save_folder_path.get():
            self.log("Warning: Main app's output folder is not set. Processing will likely fail.")

    def _build_ui(self) -> None:
        """Creates all UI elements."""
        self.grid_columnconfigure(0, weight=1, minsize=350)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        left_panel = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        left_panel.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1);
        left_panel.grid_rowconfigure(4, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left_panel, text="Source EEG Files", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=PAD_X, pady=PAD_Y, sticky="w")
        self.source_files_listbox = create_themed_listbox(left_panel, selectmode=tk.EXTENDED, exportselection=False)
        self.source_files_listbox.grid(row=1, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")

        source_buttons_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        source_buttons_frame.grid(row=2, column=0, pady=PAD_Y, sticky="ew")
        ctk.CTkButton(source_buttons_frame, text="Add Files...", command=self.add_source_files,
                      width=BUTTON_WIDTH).pack(side="left", padx=PAD_X, expand=True)
        ctk.CTkButton(source_buttons_frame, text="Remove Selected", command=self.remove_source_files,
                      width=BUTTON_WIDTH).pack(side="left", padx=PAD_X, expand=True)

        ctk.CTkLabel(left_panel, text="Defined Averaging Groups", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=PAD_X, pady=(PAD_Y * 2, PAD_Y), sticky="w")
        self.groups_listbox = create_themed_listbox(left_panel, exportselection=False)
        self.groups_listbox.grid(row=4, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        self.groups_listbox.bind("<<ListboxSelect>>", self.on_group_select)

        group_buttons_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        group_buttons_frame.grid(row=5, column=0, pady=PAD_Y, sticky="ew")
        ctk.CTkButton(group_buttons_frame, text="Create New Group", command=self.create_new_group,
                      width=BUTTON_WIDTH).pack(side="left", padx=PAD_X, expand=True)
        ctk.CTkButton(group_buttons_frame, text="Delete Group", command=self.delete_selected_group,
                      width=BUTTON_WIDTH).pack(side="left", padx=PAD_X, expand=True)

        right_panel = ctk.CTkFrame(self, corner_radius=CORNER_RADIUS)
        right_panel.grid(row=0, column=1, padx=(0, PAD_X), pady=PAD_Y, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1);
        right_panel.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(right_panel, text="Group Configuration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0,
                                                                                                    padx=PAD_X,
                                                                                                    pady=PAD_Y,
                                                                                                    sticky="w")
        self.group_config_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.group_config_frame.grid(row=1, column=0, padx=PAD_X, pady=PAD_Y, sticky="new")
        self.group_config_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right_panel, text="Condition Mapping for Selected Group", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=PAD_X, pady=(PAD_Y * 2, PAD_Y), sticky="w")
        self.condition_mapping_frame = ctk.CTkScrollableFrame(right_panel,
                                                              label_text="Define how conditions are averaged for this group")
        self.condition_mapping_frame.grid(row=3, column=0, padx=PAD_X, pady=PAD_Y, sticky="nsew")
        if hasattr(self.condition_mapping_frame, '_scrollbar') and self.condition_mapping_frame._scrollbar:
            self.condition_mapping_frame._scrollbar.configure(height=0)
        self.condition_mapping_frame.grid_columnconfigure(0, weight=1)

        avg_method_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        avg_method_frame.grid(row=4, column=0, padx=PAD_X, pady=PAD_Y, sticky="ew")
        ctk.CTkLabel(avg_method_frame, text="Averaging Method:").pack(side="left", padx=PAD_X)
        self.averaging_method_var = tk.StringVar(value="Pool Trials")
        ctk.CTkRadioButton(avg_method_frame, text="Pool Trials", variable=self.averaging_method_var,
                           value="Pool Trials", command=self._update_current_group_avg_method).pack(side="left",
                                                                                                    padx=PAD_X)
        ctk.CTkRadioButton(avg_method_frame, text="Average of Averages", variable=self.averaging_method_var,
                           value="Average of Averages", command=self._update_current_group_avg_method).pack(side="left",
                                                                                                            padx=PAD_X)

        self.save_group_config_button = ctk.CTkButton(right_panel, text="Save Group Configuration",
                                                      command=self.save_current_group_config,
                                                      width=int(BUTTON_WIDTH * 1.2))
        self.save_group_config_button.grid(row=5, column=0, padx=PAD_X, pady=PAD_Y * 2, sticky="e")
        self.save_group_config_button.configure(state="disabled")

        bottom_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        bottom_panel.grid(row=1, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y, sticky="ew")
        bottom_panel.grid_columnconfigure(0, weight=1)

        self.log_textbox = ctk.CTkTextbox(bottom_panel, height=100, wrap="word", state="disabled")
        self.log_textbox.grid(row=0, column=0, columnspan=2, padx=PAD_X, pady=PAD_Y, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(bottom_panel, orientation="horizontal")
        self.progress_bar.grid(row=1, column=0, columnspan=2, padx=PAD_X, pady=(0, PAD_Y), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()  # Hide initially

        control_buttons_frame = ctk.CTkFrame(bottom_panel, fg_color="transparent")
        control_buttons_frame.grid(row=2, column=0, columnspan=2, pady=PAD_Y, sticky="e")
        self.start_adv_processing_button = ctk.CTkButton(control_buttons_frame, text="Start Advanced Processing",
                                                         command=self.start_advanced_processing,
                                                         font=ctk.CTkFont(weight="bold"))
        self.start_adv_processing_button.pack(side="left", padx=PAD_X)
        self.close_button = ctk.CTkButton(control_buttons_frame, text="Close", command=self._on_close)
        self.close_button.pack(side="left", padx=PAD_X)
        self._clear_group_config_display()

    def _center_window(self) -> None:
        self.update_idletasks()
        self.geometry(
            f'{self.winfo_width()}x{self.winfo_height()}+{(self.winfo_screenwidth() - self.winfo_width()) // 2}+{(self.winfo_screenheight() - self.winfo_height()) // 2}')

    def log(self, message: str) -> None:
        if self.log_textbox.winfo_exists():  # Check if widget still exists
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert(tk.END, f"{message}\n")
            self.log_textbox.see(tk.END)
            self.log_textbox.configure(state="disabled")
        if hasattr(self.master_app, 'log'):
            self.master_app.log(f"[AdvAnalysis] {message}")

    def add_source_files(self) -> None:
        ftypes = [("EEG files", "*.bdf *.set"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(title="Select Source EEG Files", filetypes=ftypes, parent=self)
        if files:
            new_count = sum(1 for f_path in files if f_path not in self.source_eeg_files)
            self.source_eeg_files.extend(f for f in files if f not in self.source_eeg_files)
            if new_count > 0:
                self.source_eeg_files.sort()
                self._update_source_files_listbox()
                self.log(f"Added {new_count} source file(s). Total: {len(self.source_eeg_files)}.")

    def remove_source_files(self) -> None:
        sel_indices = self.source_files_listbox.curselection()
        if not sel_indices: self.log("No source files selected to remove."); return
        removed_paths = [self.source_eeg_files[i] for i in sel_indices]
        self.source_eeg_files = [f for i, f in enumerate(self.source_eeg_files) if i not in sel_indices]
        if removed_paths:
            self._update_source_files_listbox()
            self.log(f"Removed {len(removed_paths)} file(s) from source list.")
            self._check_groups_for_removed_files(removed_paths)

    def _check_groups_for_removed_files(self, removed_paths: List[str]) -> None:
        updated_any = False
        affected_group_indices = []
        for idx, group in enumerate(self.defined_groups):
            original_files_count = len(group['file_paths'])
            group['file_paths'] = [fp for fp in group['file_paths'] if fp not in removed_paths]
            if len(group['file_paths']) != original_files_count:
                self.log(f"Group '{group['name']}' updated: removed files.")
                updated_any = True;
                affected_group_indices.append(idx)
                group['config_saved'] = False  # Requires re-save
                if 'condition_mappings' in group:
                    for rule in group['condition_mappings']:
                        original_src_count = len(rule['sources'])
                        rule['sources'] = [s for s in rule['sources'] if s['file_path'] not in removed_paths]
                        if not rule['sources'] and original_src_count > 0:
                            msg = f"Mapping '{rule['output_label']}' in group '{group['name']}' lost all sources and was removed."
                            self.log(f"Warning: {msg}")
                            tk_messagebox.showwarning("Mapping Rule Removed", msg, parent=self)
                    group['condition_mappings'] = [r for r in group['condition_mappings'] if r['sources']]
        if updated_any:
            self._update_groups_listbox()
            if self.selected_group_index is not None and self.selected_group_index in affected_group_indices:
                self._display_group_configuration()  # Refresh if current group affected
            self._update_start_processing_button_state()

    def _update_source_files_listbox(self) -> None:
        self.source_files_listbox.delete(0, tk.END)
        for f_path in self.source_eeg_files: self.source_files_listbox.insert(tk.END, os.path.basename(f_path))

    def create_new_group(self) -> None:
        dialog = ctk.CTkInputDialog(text="Enter name for the new averaging group:", title="Create Group")
        name = dialog.get_input()
        if not name or not name.strip(): self.log("Group creation cancelled."); return
        name = name.strip()
        if any(g['name'] == name for g in self.defined_groups):
            tk_messagebox.showerror("Error", f"Group '{name}' already exists.", parent=self);
            return
        sel_indices = self.source_files_listbox.curselection()
        if not sel_indices:
            tk_messagebox.showwarning("No Files", "Select source files to add to the new group.", parent=self);
            return

        new_group = {'name': name, 'file_paths': [self.source_eeg_files[i] for i in sel_indices],
                     'condition_mappings': [], 'averaging_method': self.averaging_method_var.get(),
                     'ui_mapping_rules': [], 'config_saved': False}
        self.defined_groups.append(new_group)
        self._update_groups_listbox();
        self.log(f"Created group '{name}'. Configure and save it.")
        self.groups_listbox.selection_clear(0, tk.END)
        self.groups_listbox.selection_set(len(self.defined_groups) - 1)
        self.on_group_select(None)
        self._update_start_processing_button_state()

    def delete_selected_group(self) -> None:
        if self.selected_group_index is None: self.log("No group selected to delete."); return
        name = self.defined_groups[self.selected_group_index]['name']
        if tk_messagebox.askyesno("Confirm Delete", f"Delete group '{name}'?", parent=self):
            del self.defined_groups[self.selected_group_index]
            self._update_groups_listbox();
            self.log(f"Deleted group: {name}")
            self.selected_group_index = None
            self._clear_group_config_display()
            self._update_start_processing_button_state()

    def _update_groups_listbox(self) -> None:
        sel_idx = self.groups_listbox.curselection()
        self.groups_listbox.delete(0, tk.END)
        for group in self.defined_groups:
            status = "" if group.get('config_saved', False) else " (Unsaved)"
            self.groups_listbox.insert(tk.END, f"{group['name']} ({len(group['file_paths'])} files){status}")
        if sel_idx and sel_idx[0] < self.groups_listbox.size():
            self.groups_listbox.selection_set(sel_idx[0]);
            self.groups_listbox.activate(sel_idx[0])

    def on_group_select(self, event: Optional[tk.Event]) -> None:
        sel_indices = self.groups_listbox.curselection()
        if not sel_indices: return
        new_idx = sel_indices[0]
        if self.selected_group_index != new_idx:
            self.selected_group_index = new_idx
            self.log(f"Selected group: {self.defined_groups[self.selected_group_index]['name']}")
        self._display_group_configuration()
        self.save_group_config_button.configure(state="normal")

    def _clear_group_config_display(self) -> None:
        for widget in self.group_config_frame.winfo_children(): widget.destroy()
        for widget in self.condition_mapping_frame.winfo_children(): widget.destroy()
        ctk.CTkLabel(self.condition_mapping_frame, text="Select or create a group to configure mappings.").pack(
            padx=PAD_X, pady=PAD_Y)
        self.save_group_config_button.configure(state="disabled")
        self.averaging_method_var.set("Pool Trials")

    def _display_group_configuration(self) -> None:
        self._clear_group_config_display()
        if self.selected_group_index is None or self.selected_group_index >= len(self.defined_groups): return
        group = self.defined_groups[self.selected_group_index]
        ctk.CTkLabel(self.group_config_frame, text=f"Group Name: {group['name']}",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=PAD_X, pady=PAD_Y)
        files_text = "\n".join([os.path.basename(fp) for fp in group['file_paths']]) or "No files in group."
        ctk.CTkLabel(self.group_config_frame, text="Files in this group:\n" + files_text, justify=tk.LEFT,
                     anchor="w").pack(anchor="w", padx=PAD_X)

        for widget in self.condition_mapping_frame.winfo_children(): widget.destroy()
        ctk.CTkButton(self.condition_mapping_frame, text="+ Add Condition Mapping Rule",
                      command=self._add_condition_mapping_row_ui).pack(pady=PAD_Y, padx=PAD_X, anchor="nw")

        group['ui_mapping_rules'] = []
        for mapping_item in group.get('condition_mappings', []): self._recreate_mapping_row_ui(mapping_item)
        self.averaging_method_var.set(group.get('averaging_method', "Pool Trials"))
        self.save_group_config_button.configure(state="normal")

    def _add_condition_mapping_row_ui(self, mapping_to_edit: Optional[Dict[str, Any]] = None) -> None:
        if self.selected_group_index is None: return
        group = self.defined_groups[self.selected_group_index]
        # Ensure frame for rule has a unique border color for error highlighting
        rule_frame_border_color = ctk.ThemeManager.theme["CTkFrame"]["border_color"][ctk.get_appearance_mode()]
        rule_frame = ctk.CTkFrame(self.condition_mapping_frame, border_width=1, border_color=rule_frame_border_color,
                                  corner_radius=CORNER_RADIUS - 2)
        rule_frame.pack(fill="x", expand=True, padx=PAD_X, pady=PAD_Y)

        new_label_frame = ctk.CTkFrame(rule_frame, fg_color="transparent")
        new_label_frame.pack(fill="x", padx=PAD_X, pady=PAD_Y // 2)
        ctk.CTkLabel(new_label_frame, text="New Averaged Label:").pack(side="left", padx=(0, PAD_X))
        new_label_entry = ctk.CTkEntry(new_label_frame, placeholder_text="e.g., Combined_Faces",
                                       width=ADV_LABEL_ID_ENTRY_WIDTH)
        new_label_entry.pack(side="left", fill="x", expand=True)
        if mapping_to_edit: new_label_entry.insert(0, mapping_to_edit.get('output_label', ''))

        sources_outer_frame = ctk.CTkFrame(rule_frame, fg_color="transparent")
        sources_outer_frame.pack(fill="x", expand=True, padx=PAD_X, pady=PAD_Y // 2)

        source_widgets = []
        files_for_ui = group['file_paths'] if not mapping_to_edit else [src['file_path'] for src in
                                                                        mapping_to_edit.get('sources', [])]

        for file_path in files_for_ui:
            if file_path not in self.source_eeg_files: continue  # Skip if globally removed
            file_map_frame = ctk.CTkFrame(sources_outer_frame, fg_color="transparent")
            file_map_frame.pack(fill="x", pady=1)
            fname_disp = os.path.basename(file_path);
            fname_disp = fname_disp[:10] + "..." + fname_disp[-12:] if len(fname_disp) > 25 else fname_disp
            ctk.CTkLabel(file_map_frame, text=f"{fname_disp}:", width=100, anchor="w").pack(side="left",
                                                                                            padx=(0, PAD_X))
            orig_label_e = ctk.CTkEntry(file_map_frame, placeholder_text="Orig. Label", width=ADV_ENTRY_WIDTH);
            orig_label_e.pack(side="left", padx=PAD_X, fill="x", expand=True)
            ctk.CTkLabel(file_map_frame, text="ID:").pack(side="left", padx=(PAD_X * 2, 0))
            orig_id_e = ctk.CTkEntry(file_map_frame, placeholder_text="ID", width=ADV_ID_ENTRY_WIDTH);
            orig_id_e.pack(side="left", padx=PAD_X)

            src_data = next((s for s in mapping_to_edit.get('sources', []) if s.get('file_path') == file_path),
                            None) if mapping_to_edit else None
            if src_data: orig_label_e.insert(0, src_data.get('original_label', '')); orig_id_e.insert(0,
                                                                                                      str(src_data.get(
                                                                                                          'original_id',
                                                                                                          '')))
            source_widgets.append({'file_path': file_path, 'label_entry': orig_label_e, 'id_entry': orig_id_e})

        ctk.CTkButton(new_label_frame, text="✕", width=28, height=28, corner_radius=CORNER_RADIUS,
                      command=lambda rf=rule_frame, mte=mapping_to_edit: self._remove_mapping_rule_ui_and_data(rf,
                                                                                                               mte)).pack(
            side="right", padx=(PAD_X * 2, 0))

        group.setdefault('ui_mapping_rules', []).append(
            {'frame': rule_frame, 'new_label_entry': new_label_entry, 'sources_widgets': source_widgets,
             '_data_ref': mapping_to_edit, '_default_border_color': rule_frame_border_color})
        if not mapping_to_edit: self.log(f"Added UI row for condition mapping in group '{group['name']}'.")

    def _recreate_mapping_row_ui(self, mapping_item: Dict[str, Any]) -> None:
        self._add_condition_mapping_row_ui(mapping_to_edit=mapping_item)

    def _remove_mapping_rule_ui_and_data(self, frame_to_remove: ctk.CTkFrame,
                                         data_ref: Optional[Dict[str, Any]]) -> None:
        if self.selected_group_index is None: return
        group = self.defined_groups[self.selected_group_index]
        group['ui_mapping_rules'] = [ui_r for ui_r in group.get('ui_mapping_rules', []) if
                                     ui_r['frame'] != frame_to_remove]
        if data_ref and 'condition_mappings' in group:
            try:
                group['condition_mappings'].remove(data_ref); self.log("Removed saved mapping rule data."); group[
                    'config_saved'] = False; self._update_groups_listbox(); self._update_start_processing_button_state()
            except ValueError:
                self.log("Warning: Could not find exact mapping data ref to remove.")
        frame_to_remove.destroy();
        self.log("Removed condition mapping UI row.")

    def _highlight_error_row(self, ui_rule_info: Dict[str, Any],
                             error_field_widget: Optional[ctk.CTkEntry] = None) -> None:
        """Highlights a mapping rule row with an error."""
        error_color = "red"  # Or a theme-appropriate error color
        ui_rule_info['frame'].configure(border_color=error_color)
        if error_field_widget: error_field_widget.focus_set()
        # Schedule reset of border color after a delay
        self.after(3000, lambda: ui_rule_info['frame'].configure(border_color=ui_rule_info['_default_border_color']))

    def save_current_group_config(self) -> bool:
        if self.selected_group_index is None: self.log("No group selected."); return False
        group = self.defined_groups[self.selected_group_index]
        group['averaging_method'] = self.averaging_method_var.get()

        collected_mappings = [];
        output_labels = set()
        if not group.get('ui_mapping_rules'):  # No UI rules, save as 0 mappings
            group['condition_mappings'] = [];
            group['config_saved'] = True
            self.log(f"Config saved for '{group['name']}' (0 mappings).");
            self._update_groups_listbox();
            self._update_start_processing_button_state()
            tk_messagebox.showinfo("Config Saved", f"Config for '{group['name']}' (0 mappings) saved.", parent=self);
            return True

        for ui_rule in group['ui_mapping_rules']:
            new_label = ui_rule['new_label_entry'].get().strip()
            if not new_label: tk_messagebox.showerror("Error", "New Label empty.",
                                                      parent=self); self._highlight_error_row(ui_rule, ui_rule[
                'new_label_entry']); return False
            if new_label in output_labels: tk_messagebox.showerror("Error", f"Duplicate New Label: '{new_label}'.",
                                                                   parent=self); self._highlight_error_row(ui_rule,
                                                                                                           ui_rule[
                                                                                                               'new_label_entry']); return False
            output_labels.add(new_label)

            mapping_data = {'output_label': new_label, 'sources': []};
            has_one_src = False
            for src_widget_info in ui_rule['sources_widgets']:
                label, id_str = src_widget_info['label_entry'].get().strip(), src_widget_info['id_entry'].get().strip()
                if label and id_str:
                    has_one_src = True
                    try:
                        id_val = int(id_str)
                    except ValueError:
                        tk_messagebox.showerror("Error",
                                                f"ID '{id_str}' for '{label}' in {os.path.basename(src_widget_info['file_path'])} not num.",
                                                parent=self); self._highlight_error_row(ui_rule, src_widget_info[
                            'id_entry']); return False
                    mapping_data['sources'].append(
                        {'file_path': src_widget_info['file_path'], 'original_label': label, 'original_id': id_val})
                elif label or id_str:
                    tk_messagebox.showerror("Error",
                                            f"In {os.path.basename(src_widget_info['file_path'])}, provide both Label & ID or neither.",
                                            parent=self); self._highlight_error_row(ui_rule, src_widget_info[
                        'label_entry'] if not label else src_widget_info['id_entry']); return False
            if not has_one_src: tk_messagebox.showerror("Error", f"Rule for '{new_label}' needs >=1 source.",
                                                        parent=self); self._highlight_error_row(ui_rule); return False
            collected_mappings.append(mapping_data)

        group['condition_mappings'] = collected_mappings;
        group['config_saved'] = True
        self.log(f"Config saved for '{group['name']}' ({len(collected_mappings)} rules).");
        self._update_groups_listbox();
        self._update_start_processing_button_state()
        tk_messagebox.showinfo("Config Saved", f"Config for '{group['name']}' saved.", parent=self);
        return True

    def _update_current_group_avg_method(self) -> None:
        if self.selected_group_index is not None and self.selected_group_index < len(self.defined_groups):
            group = self.defined_groups[self.selected_group_index]
            if group.get('averaging_method') != self.averaging_method_var.get():
                group['averaging_method'] = self.averaging_method_var.get();
                group['config_saved'] = False
                self.log(f"Avg method for '{group['name']}' changed. Needs re-saving.");
                self._update_groups_listbox();
                self._update_start_processing_button_state()

    def _update_start_processing_button_state(self) -> None:
        state = "normal" if self.defined_groups and all(
            g.get('config_saved') for g in self.defined_groups) else "disabled"
        self.start_adv_processing_button.configure(state=state)

    def start_advanced_processing(self) -> None:
        self.log("=" * 30 + "\nAttempting to Start Advanced Processing...")
        if not self.defined_groups or not all(g.get('config_saved') for g in self.defined_groups):
            tk_messagebox.showerror("Error", "Define and save all groups first.", parent=self);
            return
        if not hasattr(self.master_app, 'validated_params') or not self.master_app.validated_params or \
                not hasattr(self.master_app, 'save_folder_path') or not self.master_app.save_folder_path.get():
            tk_messagebox.showerror("Error", "Main app params/output folder not set.", parent=self);
            return

        self.log("All configs validated. Starting processing thread...");
        self.progress_bar.grid();
        self.progress_bar.set(0)
        self.start_adv_processing_button.configure(state="disabled");
        self.close_button.configure(state="disabled")
        self._stop_requested.clear()
        self.processing_thread = threading.Thread(target=self._processing_thread_func, daemon=True)
        self.processing_thread.start()
        self.after(100, self._check_processing_thread)

    def _check_processing_thread(self) -> None:
        """Periodically checks the status of the processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.after(100, self._check_processing_thread)  # Keep checking
        else:  # Thread finished or was never started properly
            # Final UI updates are typically handled by the thread itself via self.after
            # This is more of a fallback or for cleanup if thread ended unexpectedly.
            if not self._stop_requested.is_set():  # If not stopped by user, means it finished or crashed
                # If _finalize_processing_ui wasn't called by thread, call it.
                # This depends on how thread signals completion.
                pass
            self.log("Processing thread has finished.")
            self._finalize_processing_ui_state()  # Ensure UI is reset

    def _processing_thread_func(self) -> None:
        """Core MNE processing logic will go here, run in a separate thread."""
        # This is a placeholder simulation. Replace with actual MNE calls.
        # --- Start of Actual MNE logic (to be implemented) ---
        # Access main app parameters:
        # main_app_params = self.master_app.validated_params.copy()
        # output_dir = self.master_app.save_folder_path.get()
        # stim_channel_name = main_app_params.get('stim_channel', self.master_app.DEFAULT_STIM_CHANNEL)
        # --- End of Actual MNE logic ---

        total_groups = len(self.defined_groups)
        # Estimate total steps: groups + mapping rules per group + sources per rule + post-proc per group
        total_ops = sum(1 + len(g.get('condition_mappings', [])) + sum(
            len(m.get('sources', [])) for m in g.get('condition_mappings', [])) + 1 for g in self.defined_groups)
        current_op = 0

        try:
            for group_idx, group_data in enumerate(self.defined_groups):
                if self._stop_requested.is_set(): self.after(0,
                                                             lambda: self.log("Processing cancelled by user.")); break
                self.after(0, lambda g=group_data['name']: self.log(f"Processing group: {g}"))
                # --- Actual MNE: group setup ---
                time.sleep(0.2)  # Simulate
                current_op += 1
                self.after(0, lambda val=current_op / total_ops: self.progress_bar.set(val))

                pid = self._extract_pid_for_group(group_data)
                self.after(0, lambda p=pid: self.log(f"  PID for group: {p}"))
                # averaged_epochs_for_group = {} # Store {output_label: MNE_Epochs/Evoked}

                for map_rule in group_data.get('condition_mappings', []):
                    if self._stop_requested.is_set(): break
                    self.after(0, lambda ol=map_rule['output_label']: self.log(f"  Processing mapping rule: {ol}"))
                    # --- Actual MNE: mapping rule setup ---
                    time.sleep(0.1)  # Simulate
                    # epochs_from_sources = []
                    for source in map_rule.get('sources', []):
                        if self._stop_requested.is_set(): break
                        self.after(0, lambda sp=os.path.basename(source['file_path']), sl=source['original_label'],
                                             si=source['original_id']: self.log(
                            f"    Load/Preproc/Epoch: {sp} ({sl}:{si})"))
                        # --- Actual MNE: Load, Preprocess, Epoch for one source ---
                        # raw = self.master_app.load_eeg_file(source['file_path'])
                        # if self._stop_requested.is_set() or not raw: continue
                        # raw_proc = self.master_app.preprocess_raw(raw.copy(), **main_app_params)
                        # if self._stop_requested.is_set() or not raw_proc: continue
                        # events = mne.find_events(raw_proc, stim_channel=stim_channel_name, ...)
                        # current_source_epochs = mne.Epochs(raw_proc, events, event_id={source['original_label']: source['original_id']}, ...)
                        # epochs_from_sources.append(current_source_epochs)
                        time.sleep(0.3)  # Simulate file processing
                        current_op += 1
                        self.after(0, lambda val=current_op / total_ops: self.progress_bar.set(val))
                    if self._stop_requested.is_set(): break

                    self.after(0, lambda ol=map_rule['output_label'], am=group_data['averaging_method']: self.log(
                        f"    Averaging for '{ol}' using {am}"))
                    # --- Actual MNE: Averaging based on group_data['averaging_method'] ---
                    # if group_data['averaging_method'] == "Pool Trials" and epochs_from_sources:
                    #    concatenated = mne.concatenate_epochs(epochs_from_sources)
                    #    averaged_result = concatenated.average() # Evoked
                    #    averaged_epochs_for_group[map_rule['output_label']] = [averaged_result] # post_process expects list
                    # elif ... "Average of Averages" ...
                    time.sleep(0.2)  # Simulate
                    current_op += 1
                    self.after(0, lambda val=current_op / total_ops: self.progress_bar.set(val))
                if self._stop_requested.is_set(): break

                self.after(0, lambda gn=group_data['name'], p=pid: self.log(
                    f"  Post-processing & Excel for group '{gn}' (PID: {p})"))
                # --- Actual MNE: Call adapted post_process ---
                # if averaged_epochs_for_group:
                #    post_process_adapted(app_context, list(averaged_epochs_for_group.keys()), averaged_epochs_for_group, pid, output_dir, group_data['name'])
                time.sleep(0.5)  # Simulate
                current_op += 1
                self.after(0, lambda val=current_op / total_ops: self.progress_bar.set(val))
            if self._stop_requested.is_set(): return  # Early exit from thread

            self.after(0, lambda: self.log("Advanced processing successfully completed!"))
            self.after(0,
                       lambda: tk_messagebox.showinfo("Processing Complete", "Advanced averaging process has finished.",
                                                      parent=self))
        except Exception as e:
            self.after(0, lambda err=str(e): self.log(f"ERROR in processing thread: {err}\n{traceback.format_exc()}"))
            self.after(0, lambda err=str(e): tk_messagebox.showerror("Processing Error", f"An error occurred: {err}",
                                                                     parent=self))
        finally:
            self.after(0, self._finalize_processing_ui_state)  # Ensure UI is reset from main thread

    def _finalize_processing_ui_state(self) -> None:
        """Resets UI elements after processing is done or stopped."""
        self.progress_bar.set(0);
        self.progress_bar.grid_remove()
        self.close_button.configure(state="normal")
        self._update_start_processing_button_state()  # Re-evaluates if start button should be enabled
        if not self._stop_requested.is_set():  # Only if not cancelled
            self.log("Ready for next advanced analysis.")
        self._stop_requested.clear()  # Reset for next run
        self.processing_thread = None

    def _extract_pid_for_group(self, group_data: Dict[str, Any]) -> str:
        if not group_data.get('file_paths'): return "UnknownPID"
        pid_base = os.path.splitext(os.path.basename(group_data['file_paths'][0]))[0]
        # Consider making this regex configurable via config.py
        match = re.search(r'\b(P\d+|Sub\d+|S\d+)\b', pid_base, re.IGNORECASE)
        if match: return match.group(1).upper()
        pid_cleaned = re.sub(r'(_unamb|_ambig|_mid|_run\d*|_sess\d*|_task\w*|_eeg|_fpvs|_raw|_preproc|_ica).*$', '',
                             pid_base, flags=re.IGNORECASE)
        pid_cleaned = re.sub(r'[^a-zA-Z0-9]', '', pid_cleaned)
        return pid_cleaned if pid_cleaned else pid_base

    def _on_close(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            if tk_messagebox.askyesno("Confirm Close", "Processing is ongoing. Stop and close?", parent=self):
                self._stop_requested.set()  # Signal thread to stop
                self.log("Stop requested. Waiting for thread to terminate...")
                # Give thread a moment to stop, then destroy.
                # A more robust solution might involve thread.join(timeout) here.
                self.after(500, self._force_destroy)  # Schedule destroy if thread doesn't stop quickly
            else:
                return  # Don't close
        else:
            self.grab_release();
            self.destroy()

    def _force_destroy(self) -> None:
        """Forcefully destroys the window after attempting graceful shutdown."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.log("Thread still active. Forcing close.")
        self.grab_release()
        self.destroy()


if __name__ == '__main__':
    import traceback  # For main's exception logging

    root = ctk.CTk()


    class DummyMasterApp(ctk.CTk):
        def __init__(self):
            super().__init__();
            self.withdraw()
            self.validated_params = {'low_pass': 0.1, 'high_pass': 50, 'epoch_start': -1, 'epoch_end': 2,
                                     'stim_channel': 'Status'}
            self.save_folder_path = tk.StringVar(value=os.path.join(os.getcwd(), "test_adv_output"))
            try:
                os.makedirs(self.save_folder_path.get(), exist_ok=True)
            except Exception as e:
                print(f"Error creating dummy output dir: {e}")
            self.DEFAULT_STIM_CHANNEL = 'Status'

        def log(self, message):
            print(f"[DummyMasterLog] {message}")

        def load_eeg_file(self, filepath):
            print(f"DummyLoad: {filepath}"); return "dummy_raw_obj"

        def preprocess_raw(self, raw, **params):
            print(f"DummyPreproc with {params}"); return "dummy_proc_raw_obj"


    try:
        dummy_master = DummyMasterApp()
        app_adv = AdvancedAnalysisWindow(master=dummy_master)
        dummy_master.mainloop()
    except Exception as e:
        print("Error in __main__:", traceback.format_exc())

